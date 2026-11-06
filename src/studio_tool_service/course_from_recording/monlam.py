"""
Tibetan transcription with Monlam AI. Ported from the pipeline's transcribe.py: pieces are sent as
Monlam jobs, PARALLEL at a time, with retries. Every reply is saved to S3 the moment it arrives, so a
job that is retried or restarted never pays for the same piece twice. A request Monlam refuses outright
(a wrong key, no credit left) is not retried: the whole step stops at once with a plain reason.
"""

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import requests
import soundfile as sf

from .pieces import SR

log = logging.getLogger(__name__)

PARALLEL = 3
POLL_SECONDS = 3
JOB_TIMEOUT = 10 * 60
RETRIES = 3
# Answers that trying again won't change: the key, the account or the request itself is refused.
REFUSED = {400, 401, 402, 403, 413, 415, 422}


class MonlamRefused(Exception):
    def __init__(self, status: int, detail: str = ""):
        super().__init__(f"Monlam refused the request (HTTP {status}) {detail}".strip())
        self.status = status


def check(resp: requests.Response) -> None:
    if resp.status_code in REFUSED:
        raise MonlamRefused(resp.status_code, resp.text[:200])
    resp.raise_for_status()


class Monlam:
    def __init__(self, base_url: str, api_key: str):
        self.base_url, self.headers = base_url.rstrip("/"), {"X-API-Key": api_key}

    def submit(self, wav: Path, idempotency_key: str) -> str:
        with wav.open("rb") as f:
            resp = requests.post(
                f"{self.base_url}/api/v1/speech-to-text/jobs",
                headers={**self.headers, "x-idempotency-key": idempotency_key},
                files={"file": (wav.name, f, "audio/wav")},
                data={"language": "bo", "task": "transcribe", "return_timestamps": "true"},
                timeout=120,
            )
        check(resp)
        return resp.json()["job_id"]

    def wait(self, job_id: str) -> dict:
        started = time.monotonic()
        while time.monotonic() - started < JOB_TIMEOUT:
            resp = requests.get(
                f"{self.base_url}/api/v1/speech-to-text/jobs/{job_id}", headers=self.headers, timeout=60
            )
            check(resp)
            job = resp.json()
            if job["status"] in ("COMPLETED", "FAILED"):
                return job
            time.sleep(POLL_SECONDS)
        raise TimeoutError(f"job {job_id} not finished after {JOB_TIMEOUT}s")


class Transcriber:
    """Sends pieces of one recording to Monlam; replies are kept in S3 under ``results_prefix``."""

    def __init__(self, api: Monlam, storage, results_prefix: str, audio: np.ndarray, key_prefix: str):
        self.api, self.storage, self.results_prefix = api, storage, results_prefix
        self.audio, self.key_prefix = audio, key_prefix

    def result_key(self, piece: dict) -> str:
        return f"{self.results_prefix}/piece_{piece['id']}.json"

    def tries_key(self, piece: dict) -> str:
        """How many of the piece's requests have failed so far, over every run of the job."""
        return f"{self.results_prefix}/piece_{piece['id']}.tries.json"

    def saved(self, piece: dict) -> Optional[dict]:
        """A saved reply for this piece, or None. Ignored if it was made for different audio."""
        job = self.storage.read_json(self.result_key(piece))
        if not job:
            return None
        made_for = job.get("piece")
        if made_for and (made_for["start"], made_for["end"]) != (piece["start"], piece["end"]):
            return None
        return job

    def transcribe(self, piece: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / f"piece_{piece['id']}.wav"
            sf.write(str(wav), self.audio[int(piece["start"] * SR) : int(piece["end"] * SR)], SR, subtype="PCM_16")
            last_error = None
            failed_before = (self.storage.read_json(self.tries_key(piece)) or {}).get("failed", 0)
            for attempt in range(1, RETRIES + 1):
                # The key is the same for the same request, even in a later run: if the service restarted
                # while Monlam was working on it, Monlam answers with that job instead of billing it again.
                # It changes only after a failure (counted in S3), since Monlam keeps answering 500 to a key
                # whose request failed.
                number = failed_before + attempt
                key = f"{self.key_prefix}-{piece['id']}-{piece['start']:.2f}-{piece['end']:.2f}-try{number}"
                try:
                    started = time.monotonic()
                    for submit_try in range(3):
                        try:
                            job_id = self.api.submit(wav, key)
                            break
                        except requests.RequestException:
                            if submit_try == 2:
                                raise
                            time.sleep(5 * (submit_try + 1))
                    job = self.api.wait(job_id)
                    if job["status"] == "COMPLETED":
                        job["seconds_taken"] = round(time.monotonic() - started, 1)
                        job["piece"] = {"start": piece["start"], "end": piece["end"]}
                        self.storage.write_json(self.result_key(piece), job)
                        return job
                    last_error = job.get("error") or "FAILED"
                except MonlamRefused:
                    raise  # trying again won't help
                except Exception as exc:  # noqa: BLE001 - network trouble, timeouts, HTTP errors
                    last_error = str(exc)
                self.storage.write_json(self.tries_key(piece), {"failed": number})
                log.warning("piece %s: attempt %s failed (%s); retrying", piece["id"], attempt, last_error)
                time.sleep(5 * attempt)
        raise RuntimeError(f"piece {piece['id']} failed after {RETRIES} attempts: {last_error}")

    def send_all(
        self,
        pieces: List[dict],
        results: Dict[str, dict],
        on_progress: Callable[[int, int, float], None] = lambda done, total, sent: None,
    ) -> List[str]:
        """
        Transcribe the pieces with no saved reply, PARALLEL at a time, into ``results``. Calls
        ``on_progress(done, total, seconds_sent)`` as replies arrive. Returns the ids that failed.
        """
        todo = []
        for piece in pieces:
            saved = self.saved(piece)
            if saved:
                results[piece["id"]] = saved
            else:
                todo.append(piece)
        done, sent, failed = len(pieces) - len(todo), 0.0, []
        on_progress(done, len(pieces), sent)
        if not todo:
            return []
        log.info("Sending %s pieces to Monlam, %s at a time", len(todo), PARALLEL)
        pool = ThreadPoolExecutor(PARALLEL)
        try:
            futures = {pool.submit(self.transcribe, piece): piece for piece in todo}
            for future in as_completed(futures):
                piece = futures[future]
                done += 1
                try:
                    results[piece["id"]] = future.result()
                except MonlamRefused:
                    raise  # every piece would be refused: stop now (the pieces not started are dropped)
                except Exception as exc:  # noqa: BLE001 - reported once all pieces are back
                    failed.append(piece["id"])
                    log.error("piece %s failed: %s", piece["id"], exc)
                    continue
                sent += piece["end"] - piece["start"]
                on_progress(done, len(pieces), sent)
        finally:
            # If on_progress stops the run (e.g. the author started over) or Monlam refuses, the pieces not
            # yet sent are dropped, so they are not paid for; the few already with Monlam finish first.
            pool.shutdown(wait=True, cancel_futures=True)
        return failed
