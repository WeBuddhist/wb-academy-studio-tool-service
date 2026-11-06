"""Settings for the service, read from environment variables that the Tutor plugin sets."""

import os
from dataclasses import dataclass
from typing import Mapping

REQUIRED = ("REDIS_URL", "QUEUE_NAME", "S3_BUCKET", "CMS_URL", "SERVICE_TOKEN")


@dataclass(frozen=True)
class Settings:
    redis_url: str
    queue: str
    s3_bucket: str
    cms_url: str
    service_token: str
    monlam_api_key: str
    monlam_base_url: str
    # Reserved for English transcription (round 2, not built yet): read now so the deployment is ready.
    gemini_api_key: str
    gemini_model: str
    max_recording_minutes: float
    cpu_threads: int

    @property
    def processing_queue(self) -> str:
        """Jobs being worked on; they go back to the queue if the service stops halfway."""
        return f"{self.queue}:processing"


def load(env: Mapping[str, str] = os.environ) -> Settings:
    missing = [name for name in REQUIRED if not env.get(name)]
    if missing:
        raise SystemExit(f"Missing settings: {', '.join(missing)}")
    return Settings(
        redis_url=env["REDIS_URL"],
        queue=env["QUEUE_NAME"],
        s3_bucket=env["S3_BUCKET"],
        cms_url=env["CMS_URL"].rstrip("/"),
        service_token=env["SERVICE_TOKEN"],
        # Only the transcription step needs these (Monlam for Tibetan, Gemini for English); it fails
        # with a clear message when the one it needs is missing.
        monlam_api_key=env.get("MONLAM_API_KEY", ""),
        monlam_base_url=env.get("MONLAM_BASE_URL", "https://api-v1.monlamai.studio").rstrip("/"),
        gemini_api_key=env.get("GEMINI_API_KEY", ""),
        gemini_model=env.get("GEMINI_MODEL", ""),
        max_recording_minutes=float(env.get("MAX_RECORDING_MINUTES", "240")),
        # The container's CPU limit: without it PyTorch starts a thread per core of the whole server.
        cpu_threads=max(1, int(float(env.get("CPU_THREADS", "2")))),
    )
