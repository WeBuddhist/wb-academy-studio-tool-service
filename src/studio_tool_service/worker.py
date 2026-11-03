"""
The main loop: take the next job from Redis, run it, repeat. One job at a time.

The sherab plugin adds jobs to the queue as "<tool>:<job id>", e.g. "course_from_recording:3f2c...".
A job being worked on sits in a second list, so if the service stops halfway it is put back in the
queue on the next start, and the tool continues from the last finished step.
"""
import logging
import signal
from typing import Callable, Dict

import redis

from . import config

log = logging.getLogger("studio_tool_service")

# Tool name -> function that runs one job of that tool. Each tool registers itself here.
TOOLS: Dict[str, Callable[[str, config.Settings], None]] = {}

WAIT_SECONDS = 5


def recover_unfinished(client, settings: config.Settings) -> None:
    """Put jobs left in progress by a previous run back at the front of the queue."""
    while True:
        item = client.lmove(settings.processing_queue, settings.queue, "RIGHT", "RIGHT")
        if item is None:
            return
        log.warning("Re-queued unfinished job %s", item)


def run(item: str, settings: config.Settings) -> None:
    tool, _, job_id = item.partition(":")
    if tool not in TOOLS or not job_id:
        log.error("Skipping %r: unknown tool or no job id", item)
        return
    log.info("Starting %s job %s", tool, job_id)
    TOOLS[tool](job_id, settings)
    log.info("Finished %s job %s", tool, job_id)


def process_next(client, settings: config.Settings) -> bool:
    """Wait a few seconds for a job and run it. Return False if none came."""
    item = client.blmove(settings.queue, settings.processing_queue, WAIT_SECONDS, "RIGHT", "LEFT")
    if item is None:
        return False
    try:
        run(item, settings)
    except Exception:  # noqa: BLE001 - one failed job must not stop the service
        log.exception("Job %s failed", item)
    client.lrem(settings.processing_queue, 1, item)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = config.load()
    # The socket must stay open longer than a blocking wait for a job, or the wait itself times out.
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=WAIT_SECONDS + 10)

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    # `docker stop` sends SIGTERM: stop between jobs. A job cut off halfway is re-queued on restart.
    signal.signal(signal.SIGTERM, stop)

    recover_unfinished(client, settings)
    log.info("Waiting for jobs on %s", settings.queue)
    while not stopping:
        process_next(client, settings)
    log.info("Stopped")


if __name__ == "__main__":
    main()
