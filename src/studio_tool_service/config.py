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
    s3_prefix: str
    cms_url: str
    service_token: str
    monlam_api_key: str
    max_recording_minutes: float

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
        s3_prefix=env.get("S3_PREFIX", "studio-tools").strip("/"),
        cms_url=env["CMS_URL"].rstrip("/"),
        service_token=env["SERVICE_TOKEN"],
        # Only the transcription step needs it; that step fails with a clear message without it.
        monlam_api_key=env.get("MONLAM_API_KEY", ""),
        max_recording_minutes=float(env.get("MAX_RECORDING_MINUTES", "240")),
    )
