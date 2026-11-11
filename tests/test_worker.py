import pytest

from studio_tool_service import config, worker
from studio_tool_service.errors import TryLater

ENV = {
    "REDIS_URL": "redis://redis:6379/3",
    "QUEUE_NAME": "studio_tool_service:jobs",
    "S3_BUCKET": "bucket",
    "CMS_URL": "http://cms:8000/",
    "SERVICE_TOKEN": "secret",
}


class FakeRedis:
    """The few list commands the worker uses, on plain Python lists (index 0 = LEFT)."""

    def __init__(self):
        self.lists = {}

    def _pop(self, name, side):
        items = self.lists.get(name, [])
        if not items:
            return None
        return items.pop(0 if side == "LEFT" else -1)

    def _push(self, name, side, item):
        items = self.lists.setdefault(name, [])
        items.insert(0 if side == "LEFT" else len(items), item)

    def lmove(self, src, dst, src_side, dst_side):
        item = self._pop(src, src_side)
        if item is not None:
            self._push(dst, dst_side, item)
        return item

    def blmove(self, src, dst, timeout, src_side, dst_side):
        return self.lmove(src, dst, src_side, dst_side)

    def lrem(self, name, count, item):
        self.lists[name].remove(item)

    def lpush(self, name, item):
        self._push(name, "LEFT", item)


@pytest.fixture
def settings():
    return config.load(ENV)


@pytest.fixture
def tools(monkeypatch):
    ran = []
    monkeypatch.setattr(worker, "TOOLS", {"course_from_recording": lambda job_id, _: ran.append(job_id)})
    return ran


def test_settings_are_read_from_the_environment(settings):
    assert settings.cms_url == "http://cms:8000"
    assert settings.processing_queue == "studio_tool_service:jobs:processing"


def test_transcription_keys_are_optional():
    settings = config.load({**ENV, "OPENROUTER_API_KEY": "key", "OPENROUTER_MODEL": "google/model"})
    assert (settings.monlam_api_key, settings.openrouter_api_key, settings.openrouter_model) == (
        "", "key", "google/model"
    )


def test_missing_settings_stop_the_service():
    with pytest.raises(SystemExit, match="S3_BUCKET"):
        config.load({**ENV, "S3_BUCKET": ""})


def test_runs_the_oldest_job_first(settings, tools):
    client = FakeRedis()
    client.lists[settings.queue] = ["course_from_recording:new", "course_from_recording:old"]

    assert worker.process_next(client, settings)
    assert tools == ["old"]
    assert client.lists[settings.processing_queue] == []


def test_a_failed_job_does_not_stop_the_service(settings, monkeypatch):
    def fail(job_id, _):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "TOOLS", {"course_from_recording": fail})
    client = FakeRedis()
    client.lists[settings.queue] = ["course_from_recording:a"]

    assert worker.process_next(client, settings)
    assert client.lists[settings.processing_queue] == []


def test_unfinished_jobs_go_back_to_the_front_of_the_queue(settings, tools):
    client = FakeRedis()
    client.lists[settings.queue] = ["course_from_recording:waiting"]
    client.lists[settings.processing_queue] = ["course_from_recording:interrupted"]

    worker.recover_unfinished(client, settings)
    worker.process_next(client, settings)

    assert tools == ["interrupted"]


def test_unknown_tools_are_skipped(settings, tools):
    client = FakeRedis()
    client.lists[settings.queue] = ["no_such_tool:a"]

    assert worker.process_next(client, settings)
    assert tools == []


def test_a_job_that_has_to_wait_goes_back_to_the_end_of_the_queue(settings, monkeypatch):
    def studio_down(job_id, _):
        raise TryLater("Studio is restarting")

    monkeypatch.setattr(worker, "TOOLS", {"course_from_recording": studio_down})
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    client = FakeRedis()
    client.lists[settings.queue] = ["course_from_recording:newer", "course_from_recording:waiting"]

    assert worker.process_next(client, settings)

    # Not lost: back in the queue, behind the job that was already waiting.
    assert client.lists[settings.queue] == ["course_from_recording:waiting", "course_from_recording:newer"]
    assert client.lists[settings.processing_queue] == []
