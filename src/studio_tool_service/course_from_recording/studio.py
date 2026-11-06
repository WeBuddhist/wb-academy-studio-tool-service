"""
The sherab plugin's API for this tool (course_from_recording/views_service.py on the Studio side).
The service trusts nothing in the queue entry but the job id; everything else is read from here.
"""

import logging
import time
from typing import Optional

import requests

from ..errors import TryLater

log = logging.getLogger(__name__)


class JobGone(Exception):
    """Studio has no such job (e.g. it was deleted), so there is nothing to do."""


class JobStopped(Exception):
    """Studio refused a report: the job is no longer running, e.g. the author started over ("cancelled")."""

    def __init__(self, job_id: str, status: Optional[str]):
        super().__init__(f"job {job_id} is {status or 'not running'}")
        self.status = status


class StudioUnavailable(TryLater):
    """Studio could not be reached, or kept answering with a server error, after a few tries."""


# Answers worth trying again: Studio busy or restarting. Anything else (e.g. 400) is a bug and is raised.
TRANSIENT = {429, 500, 502, 503, 504}
RETRY_DELAYS = (2, 5, 10)  # seconds between tries


class Studio:
    def __init__(self, cms_url: str, token: str, session: Optional[requests.Session] = None):
        self.base = f"{cms_url}/api/course-from-recording/v1/service/jobs"
        self.session = session or requests.Session()
        self.session.headers["X-Service-Token"] = token

    def send(self, method: str, job_id: str, attempts: int, **kwargs) -> requests.Response:
        """One request, tried again after a short wait while Studio is unreachable or busy."""
        url = f"{self.base}/{job_id}/"
        for attempt in range(attempts):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as error:
                problem = str(error)
            else:
                if resp.status_code not in TRANSIENT:
                    return resp
                problem = f"HTTP {resp.status_code}"
            if attempt + 1 < attempts:
                time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
        raise StudioUnavailable(f"Studio did not answer {method} for job {job_id}: {problem}")

    def job(self, job_id: str) -> dict:
        resp = self.send("GET", job_id, attempts=len(RETRY_DELAYS) + 1)
        if resp.status_code == 404:
            raise JobGone(job_id)
        resp.raise_for_status()
        return resp.json()

    def report(self, job_id: str, attempts: int = len(RETRY_DELAYS) + 1, **fields) -> None:
        """
        Status, progress, results, ... A refused report (the job was started over, deleted or is no longer
        running) stops the job; Studio being unreachable raises StudioUnavailable, so the job runs again later.
        """
        resp = self.send("PATCH", job_id, attempts=attempts, json=fields)
        if resp.status_code == 404:
            raise JobGone(job_id)
        if resp.status_code == 409:
            try:
                status = resp.json().get("status")
            except ValueError:
                status = None
            raise JobStopped(job_id, status)
        resp.raise_for_status()


class Progress:
    """
    Reports a step's progress (0-100) to Studio, at most every few seconds and only when it moved.
    ``start``/``end``: the share of the step this part covers, e.g. 0.1-1.0 after getting the recording ready.
    """

    def __init__(self, studio: Studio, job_id: str, start: float = 0.0, end: float = 1.0, every_seconds: float = 3.0):
        self.studio, self.job_id, self.every = studio, job_id, every_seconds
        self.start, self.end = start, end
        self.last_sent, self.last_time = -1, 0.0

    def __call__(self, done: float) -> None:
        percent = max(0, min(100, int((self.start + (self.end - self.start) * done) * 100)))
        now = time.monotonic()
        if percent != self.last_sent and (percent == 100 or now - self.last_time >= self.every):
            # Progress is only for the author to watch: one try, and a missed report is not worth stopping
            # the job for (the next one catches up). A refused report still stops it.
            self.last_time = now
            try:
                self.studio.report(self.job_id, attempts=1, progress=percent)
            except StudioUnavailable as error:
                log.warning("Job %s: progress report skipped (%s)", self.job_id, error)
                return
            self.last_sent = percent
