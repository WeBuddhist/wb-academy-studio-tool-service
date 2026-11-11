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
    # The AI model that writes the course from the transcript (phase 2), through OpenRouter; an OpenRouter
    # model id such as "google/gemini-2.5-flash-lite".
    openrouter_api_key: str
    openrouter_model: str
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
        # Only the steps that use them need these (Monlam to transcribe Tibetan, OpenRouter to write the
        # course); each fails with a clear message when its key is missing.
        monlam_api_key=env.get("MONLAM_API_KEY", ""),
        monlam_base_url=env.get("MONLAM_BASE_URL", "https://api-v1.monlamai.studio").rstrip("/"),
        openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
        openrouter_model=env.get("OPENROUTER_MODEL", ""),
        max_recording_minutes=float(env.get("MAX_RECORDING_MINUTES", "240")),
        # The container's CPU limit: without it PyTorch starts a thread per core of the whole server.
        cpu_threads=max(1, int(float(env.get("CPU_THREADS", "2")))),
    )
