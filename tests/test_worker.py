import pytest

from studio_tool_service import config, worker

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
    assert settings.s3_prefix == "studio-tools"
    assert settings.processing_queue == "studio_tool_service:jobs:processing"


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
