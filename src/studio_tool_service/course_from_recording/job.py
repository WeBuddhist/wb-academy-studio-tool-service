"""
Runs one recording through the "course from a recording" steps (phase 1: up to the transcript).

    More than one speaker: find the voices -> pause for the author's tags -> clean -> transcribe the teacher
    One speaker:           clean -> transcribe

Every step's result goes to S3 under the job's folder (chosen by Studio), and a step whose result is
already there is skipped, so a retried or restarted job continues where it stopped. Progress goes to
Studio after each step and every few seconds within one. If the author starts over, Studio refuses the
next report: the run stops there and deletes what it wrote since Studio deleted the job's files.
"""

import datetime
import logging
import os
import tempfile
from pathlib import Path

from .. import audio
from ..storage import Storage
from . import clean, pieces, transcript, voices
from .monlam import Monlam, MonlamRefused, Transcriber
from .studio import JobGone, JobStopped, Progress, Studio, StudioUnavailable

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.environ.get("MODEL_DIR", "/app/models"), "spkrec-ecapa-voxceleb")
SAMPLES_PER_VOICE = 3
SAMPLE_SECONDS = 20.0
RESPLIT_ROUNDS = 2  # how many times a garbled piece may be halved and resent
FIRST_SEND_SHARE = 0.9  # of the transcribing step's progress; the resend rounds share the rest
# Downloading and decoding the recording is the first part of whichever step a run starts with.
PREPARE_SHARE = 0.1
TRANSCRIBED = {"teacher"}

# Paths inside the job's folder; Studio keeps results relative to it.
ORIGINAL_FOLDER = "0-original/"
JOB_SUMMARY = "job.json"
TIMELINE = "1-voices/timeline.json"
WAVEFORM = "1-voices/waveform.json"
SAMPLE = "1-voices/samples/voice-{group}_sample-{number}.mp3"
CLEANED = "2-clean/clean.flac"
TRANSCRIPT_TXT = "3-transcript/transcript.txt"
TRANSCRIPT_JSON = "3-transcript/transcript.json"
RAW_TXT = "3-transcript/transcript-raw.txt"
RAW_JSON = "3-transcript/transcript-raw.json"
PLAN = "3-transcript/working/plan.json"
REPLIES = "3-transcript/working/results"

RUNNABLE = ("queued", "finding_speakers", "cleaning", "transcribing")

GENERIC_ERROR = "Something went wrong while processing the recording. Try again; if it stops again, ask for help."


class StepFailed(Exception):
    """A failure the author can act on; the message is shown on the tool page."""


def refusal_message(status: int) -> str:
    """Why Monlam refused, in words for the author."""
    if status in (401, 403):
        return "Monlam refused this server's key. Ask an administrator to check the Monlam key, then try again."
    if status == 402:
        return "The Monlam account has run out of credit. Once it is topped up, try again."
    return (
        f"Monlam refused part of the recording (error {status}). Try again; if it happens again, ask for help. "
        "The parts already done won't be sent (or paid for) again."
    )


def run(job_id: str, settings) -> None:
    """Entry point for the worker: one queue entry "course_from_recording:<job id>"."""
    studio = Studio(settings.cms_url, settings.service_token)
    try:
        job = studio.job(job_id)
    except JobGone:
        log.warning("Job %s no longer exists in Studio; nothing to do", job_id)
        return
    # A running status here means the service stopped halfway (e.g. a restart) and put the job back on
    # the queue: carry on with it. Anything else (waiting for the author, done, failed) is left alone.
    if job["status"] not in RUNNABLE:
        log.warning("Job %s is %s; nothing to do", job_id, job["status"])
        return
    storage = Storage(settings.s3_bucket)
    try:
        with tempfile.TemporaryDirectory(prefix="recording-") as tmp:
            RecordingJob(job, settings, studio, storage, Path(tmp)).run()
    except (JobStopped, JobGone) as stop:
        if isinstance(stop, JobStopped) and stop.status != "cancelled":
            log.warning("Job %s stopped: Studio refused its report (%s)", job_id, stop)
            return
        # The author started over while the job ran. Studio already deleted the job's files (and the
        # recording, unless kept), so remove what this run wrote since; the recording is Studio's to handle.
        deleted = storage.delete_folder(job["folder"], keep=ORIGINAL_FOLDER)
        log.info("Job %s: the author started over; stopped and deleted %s files written since", job_id, deleted)
    except StudioUnavailable:
        raise  # not the job's fault: the worker puts it back on the queue, and it continues from its last step
    except StepFailed as error:
        log.warning("Job %s failed: %s", job_id, error)
        studio.report(job_id, status="failed", error=str(error))
    except Exception:  # noqa: BLE001 - the author sees a plain message; the details are in the log
        log.exception("Job %s failed", job_id)
        studio.report(job_id, status="failed", error=GENERIC_ERROR)


class RecordingJob:
    def __init__(self, job: dict, settings, studio: Studio, storage: Storage, tmp: Path):
        self.job, self.settings, self.studio, self.storage, self.tmp = job, settings, studio, storage, tmp
        self.id, self.details = job["id"], job["details"]
        self.status = job["status"]

    def key(self, path: str) -> str:
        return f"{self.job['folder']}/{path}"

    def report(self, **fields) -> None:
        self.studio.report(self.id, **fields)

    def start_step(self, status: str) -> Progress:
        """Tell Studio the step has started; the first step of a run already did, while getting ready."""
        if status == self.status:
            return Progress(self.studio, self.id, start=PREPARE_SHARE)
        self.report(status=status, progress=0)
        self.status = status
        return Progress(self.studio, self.id)

    def first_step(self) -> str:
        if self.details.get("multiple_speakers") and not self.job["speaker_labels"]:
            return "finding_speakers"
        return "transcribing" if self.storage.exists(self.key(CLEANED)) else "cleaning"

    def check_can_finish(self) -> None:
        """Fail before any work (download, voices, the author's tags, cleaning) if transcription can't happen."""
        if self.details.get("teacher_language") != "bo":
            raise StepFailed("English transcription isn't available yet.")
        if not self.settings.monlam_api_key:
            raise StepFailed("Tibetan transcription isn't set up on this server yet (no Monlam key).")

    def run(self) -> None:
        self.check_can_finish()
        # Say straight away which step this run is on: downloading a long recording takes a while,
        # and the author should see it move, not "waiting for its turn".
        self.status = self.first_step()
        self.report(status=self.status, progress=0)
        preparing = Progress(self.studio, self.id, end=PREPARE_SHARE * 0.8)
        original = self.storage.download(
            self.job["original_key"], self.tmp / f"original{Path(self.job['original_key']).suffix}", preparing
        )
        length = audio.duration(original)
        if length > self.settings.max_recording_minutes * 60:
            raise StepFailed(
                f"The recording is {round(length / 60)} minutes long; this tool takes recordings of up to "
                f"{int(self.settings.max_recording_minutes)} minutes."
            )
        self.report(duration_seconds=round(length, 1))
        self.write_summary(length)

        speech = audio.read(audio.to_wav(original, self.tmp / "work-16k.wav", voices.SR))
        Progress(self.studio, self.id)(PREPARE_SHARE)
        if self.details.get("multiple_speakers"):
            if not self.job["speaker_labels"]:
                self.find_voices(original, speech)
                return  # paused: the author tags the voices, and the job comes back on the queue
            timeline = self.storage.read_json(self.key(TIMELINE))
            if timeline is None:
                raise StepFailed("The voices that were found are missing. Upload the recording again.")
            turns = voices.name_turns(timeline, self.job["speaker_labels"])
        else:
            turns = voices.speech_turns(speech)

        pauses = pieces.find_pauses(speech)
        self.clean(original, turns, pauses)
        self.transcribe(speech, pauses, turns)

    def write_summary(self, length: float) -> None:
        """job.json: what this folder is, readable without the database."""
        self.storage.write_json(
            self.key(JOB_SUMMARY),
            {
                "job_id": self.id,
                "course_id": self.job["course_id"],
                "original_filename": self.job["original_filename"],
                "original_file": self.job["original_key"][len(self.job["folder"]) + 1 :],
                "minutes": round(length / 60, 1),
                "details": self.details,
                "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            },
        )

    # ---------- 1. find the voices (more than one speaker) ----------

    def find_voices(self, original: Path, speech) -> None:
        progress = self.start_step("finding_speakers")
        timeline = self.storage.read_json(self.key(TIMELINE))
        if timeline is None:
            try:
                timeline = voices.find_voices(speech, MODEL_DIR, self.settings.cpu_threads, progress)
            except voices.NoSpeech:
                raise StepFailed(
                    "No one could be heard speaking in this recording. Check that the right file was uploaded."
                )
            self.storage.write_json(self.key(TIMELINE), timeline)
        if not self.storage.exists(self.key(WAVEFORM)):
            self.storage.write_json(self.key(WAVEFORM), voices.waveform(speech))
        groups = []
        for voice in voices.voice_groups(timeline):
            samples = []
            spans = voices.sample_spans(voice["longest"], SAMPLES_PER_VOICE, SAMPLE_SECONDS)
            for number, (start, length) in enumerate(spans, 1):
                path = SAMPLE.format(group=voice["group"], number=number)
                clip = audio.cut_mp3(original, self.tmp / "sample.mp3", start, length)
                self.storage.upload(clip, self.key(path), "audio/mpeg")
                samples.append(path)
            groups.append({"group": voice["group"], "minutes": voice["minutes"], "samples": samples})
        log.info("Job %s: %s voices found; waiting for the author's tags", self.id, len(groups))
        self.report(
            status="needs_speaker_labels",
            progress=100,
            speaker_groups=groups,
            results={"timeline": TIMELINE, "waveform": WAVEFORM},
        )

    # ---------- 2. clean ----------

    def clean(self, original: Path, turns: list, pauses) -> None:
        if self.storage.exists(self.key(CLEANED)):
            return
        progress = self.start_step("cleaning")
        wav = audio.to_wav(original, self.tmp / "work-44k.wav", clean.SR)
        try:
            noise = clean.noise_power(clean.noise_sample(wav, clean.quiet_spans(turns, pauses)))
        except clean.NoStatic:
            # Not a moment of quiet to learn the static from: keep the sound as it is rather than stop.
            log.warning("Job %s: no quiet stretch to learn the static from; the audio is kept as it is", self.id)
            noise = None
        cleaned = clean.clean_file(wav, self.tmp / "clean.flac", noise, progress)
        wav.unlink()
        self.storage.upload(cleaned, self.key(CLEANED), "audio/flac")
        self.report(results={"clean": CLEANED})

    # ---------- 3. transcribe the teacher ----------

    def transcribe(self, speech, pauses, turns: list) -> None:
        progress = self.start_step("transcribing")

        plan = pieces.plan_pieces(speech, pauses, turns, TRANSCRIBED)
        if not plan:
            raise StepFailed("No teacher speech was found to transcribe.")
        transcriber = Transcriber(
            Monlam(self.settings.monlam_base_url, self.settings.monlam_api_key),
            self.storage,
            self.key(REPLIES),
            speech,
            key_prefix=self.id[:8],
        )
        minutes_before = float(self.job.get("transcribed_minutes") or 0)
        results, sent_total = {}, [0.0]

        def send(batch: list, start: float, end: float) -> None:
            """Transcribe ``batch``; its progress fills the step from ``start`` to ``end`` (shares of 0-1)."""
            batch_sent = [0.0]

            def on_progress(done: int, total: int, sent: float) -> None:
                batch_sent[0] = sent  # before the report, which stops the run if the author started over
                progress(start + (end - start) * done / max(1, total))

            try:
                failed = transcriber.send_all(batch, results, on_progress)
            except MonlamRefused as refused:
                raise StepFailed(refusal_message(refused.status)) from refused
            finally:
                # Recorded even when the run stops here: what was sent has been paid for.
                sent_total[0] += batch_sent[0]
                self.report(transcribed_minutes=round(minutes_before + sent_total[0] / 60, 2))
            if failed:
                raise StepFailed(
                    f"{len(failed)} parts of the recording could not be transcribed by Monlam. Try again: "
                    "the parts already done won't be sent (or paid for) again."
                )

        # Most of the step is the first sending; garbled pieces sent again in halves fill the rest.
        send(plan, 0.0, FIRST_SEND_SHARE)
        resend_share = (1 - FIRST_SEND_SHARE) / RESPLIT_ROUNDS
        for round_number in range(RESPLIT_ROUNDS):
            garbled = [
                p
                for p in plan
                if pieces.is_garbled(p, results[p["id"]].get("result") or {})
                and p["end"] - p["start"] >= 2 * pieces.MIN_HALF
            ]
            if not garbled:
                break
            log.info("Job %s: resending %s garbled pieces in halves", self.id, len(garbled))
            new_plan, halves = [], []
            for piece in plan:
                parts = pieces.halve(piece, pauses, speech) if piece in garbled else [piece]
                new_plan += parts
                halves += parts if piece in garbled else []
            plan = new_plan
            start = FIRST_SEND_SHARE + round_number * resend_share
            send(halves, start, start + resend_share)

        self.storage.write_json(self.key(PLAN), plan)
        lines = transcript.build_lines(plan, results, turns)
        self.storage.write_json(self.key(RAW_JSON), lines)
        self.storage.write_text(
            self.key(RAW_TXT), transcript.as_text(lines, "# What Monlam returned, before spelling fixes")
        )
        fixed = transcript.corrected(lines)
        self.storage.write_json(self.key(TRANSCRIPT_JSON), fixed)
        self.storage.write_text(
            self.key(TRANSCRIPT_TXT), transcript.as_text(fixed, "# Transcript, Tibetan spelling fixed")
        )
        self.report(
            status="transcript_ready",
            progress=100,
            results={"transcript": TRANSCRIPT_TXT, "transcript_json": TRANSCRIPT_JSON, "transcript_raw": RAW_TXT},
        )
