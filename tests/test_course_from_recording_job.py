"""How a "course from a recording" job runs, with Studio, S3, ffmpeg and Monlam replaced by fakes."""

import numpy as np
import pytest

from studio_tool_service import audio, config
from studio_tool_service.course_from_recording import job as job_module
from studio_tool_service.course_from_recording import clean, suggest, voices
from studio_tool_service.course_from_recording.monlam import MonlamRefused
from studio_tool_service.course_from_recording.studio import JobStopped, StudioUnavailable
from studio_tool_service.llm import LLMError, LLMRefused

FOLDER = "studio-tools/course-from-recording/Sherab-tspc12-2026/2026-11-03_day-1_a101e248"
ENV = {
    "REDIS_URL": "redis://redis:6379/3",
    "QUEUE_NAME": "q",
    "S3_BUCKET": "b",
    "CMS_URL": "http://cms:8000",
    "SERVICE_TOKEN": "secret",
    "MONLAM_API_KEY": "monlam-key",
    "MAX_RECORDING_MINUTES": "60",
}
SR = 16000


class FakeStudio:
    def __init__(self, job):
        self.data, self.reports = job, []
        # Studio refuses every report once stop_when(report) has returned a status ("cancelled", ...).
        self.stopped = None
        self.stop_when = lambda fields: None
        self.unavailable_when = lambda fields: False

    def job(self, job_id):
        return self.data

    def report(self, job_id, attempts=None, **fields):
        if self.unavailable_when(fields):
            raise StudioUnavailable("Studio is restarting")
        self.stopped = self.stopped or self.stop_when(fields)
        if self.stopped and not (self.stopped == "cancelled" and set(fields) == {"transcribed_minutes"}):
            raise JobStopped(job_id, self.stopped)
        self.reports.append(fields)

    def statuses(self):
        return [r["status"] for r in self.reports if "status" in r]

    written = None  # what write_course was asked to write

    def write_course(self, job_id, lessons):
        if self.stopped:
            raise JobStopped(job_id, self.stopped)
        self.written = lessons
        return {"status": "course_ready", "sections": ["block-v1:section-1"]}


class FakeStorage:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def exists(self, key):
        return key in self.files

    def download(self, key, path, on_progress=lambda done: None):
        path.write_bytes(b"audio")
        on_progress(1.0)
        return path

    def upload(self, path, key, content_type="application/octet-stream"):
        self.files[key] = path.read_bytes()

    def write_text(self, key, text, content_type="text/plain"):
        self.files[key] = text

    def write_json(self, key, data):
        self.files[key] = data

    def read_json(self, key):
        return self.files.get(key)

    def delete_folder(self, folder, keep=""):
        doomed = [k for k in self.files if k.startswith(f"{folder}/") and not k.startswith(f"{folder}/{keep}")]
        for key in doomed:
            del self.files[key]
        return len(doomed)


class FakeMonlam:
    sent = []

    def __init__(self, base_url, api_key):
        pass

    refuse = None  # an HTTP status to refuse every request with

    def submit(self, wav, key):
        if FakeMonlam.refuse:
            raise MonlamRefused(FakeMonlam.refuse)
        FakeMonlam.sent.append(key)
        return key

    def wait(self, job_id):
        return {
            "status": "COMPLETED",
            "result": {"text": "ཆོས་གི་གཏམ།", "segments": [{"start": 0.0, "end": 2.0, "text": "ཆོས་གི་གཏམ།"}]},
        }


def a_job(**changes):
    job = {
        "id": "a101e248-9420-45f5-aa20-875e33df1f8a",
        "course_id": "course-v1:Sherab+tspc12+2026",
        "status": "queued",
        "original_filename": "Day 1.mp3",
        "folder": FOLDER,
        "original_key": f"{FOLDER}/0-original/day-1.mp3",
        "results": {},
        "speaker_groups": [],
        "speaker_labels": {},
        "transcribed_minutes": 0,
        "details": {"teacher": "Geshe Tenzin", "multiple_speakers": True, "teacher_language": "bo"},
    }
    job.update(changes)
    return job


def recording():
    """30 s of the teacher, a 10 s silence, 30 s of the translator - as phrases with short breaks."""

    def speaking(seconds, freq):
        t = np.arange(int(1.7 * SR)) / SR
        phrase = np.concatenate([0.3 * np.sin(2 * np.pi * freq * t), np.zeros(int(0.3 * SR))])
        return np.tile(phrase, int(seconds / 2)).astype(np.float32)

    quiet = np.zeros(10 * SR, np.float32)
    return np.concatenate([speaking(30, 220), quiet, speaking(30, 330)]) + 0.001


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "duration", lambda path: 70.0)
    monkeypatch.setattr(audio, "to_wav", lambda src, dest, rate: dest)
    monkeypatch.setattr(audio, "read", lambda path, start=0.0, end=None: recording())

    def cut(src, dest, start, length):
        dest.write_bytes(f"clip {start}".encode())
        return dest

    monkeypatch.setattr(audio, "cut_mp3", cut)
    monkeypatch.setattr(job_module, "Monlam", FakeMonlam)
    FakeMonlam.sent, FakeMonlam.refuse = [], None

    def run(job, storage=None, env=None, studio=None):
        studio, storage = studio or FakeStudio(job), storage or FakeStorage()
        monkeypatch.setattr(job_module, "Studio", lambda url, token: studio)
        monkeypatch.setattr(job_module, "Storage", lambda bucket: storage)
        job_module.run(job["id"], config.load({**ENV, **(env or {})}))
        return studio, storage

    return run


TIMELINE = [
    {"start": 0.0, "end": 30.0, "group": "1", "confidence": 0.3},
    {"start": 30.0, "end": 40.0, "group": None, "confidence": 0.0},
    {"start": 40.0, "end": 70.0, "group": "2", "confidence": 0.3},
]


def test_more_than_one_speaker_finds_the_voices_and_waits_for_the_author(fakes, monkeypatch):
    monkeypatch.setattr(voices, "find_voices", lambda *args, **kwargs: TIMELINE)

    studio, storage = fakes(a_job())

    assert studio.statuses() == ["finding_speakers", "needs_speaker_labels"]
    paused = studio.reports[-1]
    assert [(g["group"], g["minutes"]) for g in paused["speaker_groups"]] == [("1", 0.5), ("2", 0.5)]
    assert paused["speaker_groups"][0]["samples"] == ["1-voices/samples/voice-1_sample-1.mp3"]
    assert f"{FOLDER}/1-voices/samples/voice-2_sample-1.mp3" in storage.files
    assert storage.files[f"{FOLDER}/1-voices/timeline.json"] == TIMELINE
    wave = storage.files[f"{FOLDER}/1-voices/waveform.json"]
    assert len(wave["levels"]) == 1000 and wave["seconds_per_level"] == 0.07  # 70 s recording
    assert paused["results"] == {"timeline": "1-voices/timeline.json", "waveform": "1-voices/waveform.json"}
    assert storage.files[f"{FOLDER}/job.json"]["original_file"] == "0-original/day-1.mp3"
    assert FakeMonlam.sent == []


def test_after_tagging_only_the_teacher_is_transcribed(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})

    studio, storage = fakes(a_job(speaker_labels={"1": "teacher", "2": "translator"}), storage)

    assert studio.statuses() == ["transcribing", "transcript_ready"]  # cleaning was already done
    assert FakeMonlam.sent and all(key.startswith("a101e248-") for key in FakeMonlam.sent)
    lines = storage.files[f"{FOLDER}/3-transcript/transcript.json"]
    assert {line["speaker"] for line in lines if line["text"]} == {"teacher"}
    assert lines[0]["text"] == "ཆོས་ཀྱི་གཏམ།" and lines[0]["original"] == "ཆོས་གི་གཏམ།"  # spelling fixed
    assert any(line["speaker"] == "translator" and line["text"] is None for line in lines)
    assert studio.reports[-1]["results"]["transcript"] == "3-transcript/transcript.txt"
    assert any("transcribed_minutes" in r and r["transcribed_minutes"] > 0 for r in studio.reports)


def test_a_retry_does_not_pay_twice_for_pieces_already_transcribed(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    labels = {"1": "teacher", "2": "translator"}
    fakes(a_job(speaker_labels=labels), storage)
    first_run = len(FakeMonlam.sent)

    fakes(a_job(speaker_labels=labels), storage)

    assert first_run > 0 and len(FakeMonlam.sent) == first_run


def test_a_recording_over_the_limit_fails_before_anything_is_spent(fakes, monkeypatch):
    monkeypatch.setattr(audio, "duration", lambda path: 61 * 60.0)

    studio, _ = fakes(a_job())

    assert studio.statuses() == ["finding_speakers", "failed"]
    assert studio.reports[-1]["error"].startswith("The recording is 61 minutes long")
    assert FakeMonlam.sent == []


def test_english_waits_for_round_two_and_fails_before_any_work(fakes, monkeypatch):
    monkeypatch.setattr(voices, "find_voices", lambda *args, **kwargs: pytest.fail("no voices for English yet"))
    details = {"teacher": "T", "multiple_speakers": True, "teacher_language": "en"}

    studio, storage = fakes(a_job(details=details))

    # Failed straight away: no download, no voices for the author to tag, no cleaning.
    assert studio.reports == [{"status": "failed", "error": "English transcription isn't available yet."}]
    assert storage.files == {}


def test_without_a_monlam_key_the_author_is_told(fakes):
    storage = FakeStorage({f"{FOLDER}/2-clean/clean.flac": b"x"})
    details = {"teacher": "T", "multiple_speakers": False, "teacher_language": "bo"}

    studio, _ = fakes(a_job(details=details), storage, env={"MONLAM_API_KEY": ""})

    assert "no Monlam key" in studio.reports[-1]["error"]


def test_an_unexpected_error_fails_the_job_with_a_plain_message(fakes, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(voices, "find_voices", broken)

    studio, _ = fakes(a_job())

    assert studio.reports[-1] == {"status": "failed", "error": job_module.GENERIC_ERROR}


def test_a_job_that_is_not_queued_is_left_alone(fakes):
    studio, _ = fakes(a_job(status="needs_speaker_labels"))
    assert studio.reports == []


def test_the_step_shows_at_once_and_getting_the_recording_ready_counts_as_its_first_part(fakes, monkeypatch):
    monkeypatch.setattr(voices, "find_voices", lambda *args, **kwargs: TIMELINE)

    studio, _ = fakes(a_job())

    assert studio.reports[0] == {"status": "finding_speakers", "progress": 0}  # before the download
    progress = [r["progress"] for r in studio.reports if set(r) == {"progress"}]
    assert progress[:2] == [8, 10]  # downloaded, then decoded: the first tenth of the step
    assert studio.statuses() == ["finding_speakers", "needs_speaker_labels"]  # not reported twice


def test_a_job_interrupted_by_a_restart_carries_on(fakes, monkeypatch):
    monkeypatch.setattr(voices, "find_voices", lambda *args, **kwargs: TIMELINE)

    studio, _ = fakes(a_job(status="finding_speakers"))

    assert studio.statuses()[-1] == "needs_speaker_labels"


ORIGINAL = f"{FOLDER}/0-original/day-1.mp3"


def test_when_the_author_starts_over_the_job_stops_and_removes_what_it_wrote(fakes, monkeypatch):
    def cleaned(wav, dest, noise, on_progress):
        dest.write_bytes(b"clean")
        return dest

    monkeypatch.setattr(audio, "to_wav", lambda src, dest, rate: dest.write_bytes(b"") or dest)
    monkeypatch.setattr(clean, "noise_sample", lambda *args: (_ for _ in ()).throw(clean.NoStatic()))
    monkeypatch.setattr(clean, "clean_file", cleaned)
    details = {"teacher": "T", "multiple_speakers": False, "teacher_language": "bo"}
    job = a_job(details=details)
    studio = FakeStudio(job)
    # The author starts over while the audio is cleaned; the clean file is uploaded just before.
    studio.stop_when = lambda fields: "results" in fields and "cancelled"
    storage = FakeStorage({ORIGINAL: b"audio"})

    fakes(job, storage, studio=studio)

    assert studio.statuses() == ["cleaning"]  # no "failed", no transcribing
    assert FakeMonlam.sent == []
    assert list(storage.files) == [ORIGINAL]  # job.json and the clean file are gone; the recording is Studio's


def test_starting_over_during_transcription_stops_sending_and_keeps_the_minutes_paid_for(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    job = a_job(speaker_labels={"1": "teacher", "2": "translator"})
    studio = FakeStudio(job)
    studio.stop_when = lambda fields: (
        fields.get("progress", 0) > 10 and studio.statuses()[-1:] == ["transcribing"] and "cancelled"
    )

    fakes(job, storage, studio=studio)

    assert "transcript_ready" not in studio.statuses()
    assert studio.reports[-1]["transcribed_minutes"] > 0
    assert not any(key.startswith(f"{FOLDER}/3-transcript") for key in storage.files)


def test_a_report_refused_for_another_reason_stops_without_deleting(fakes, monkeypatch):
    monkeypatch.setattr(voices, "find_voices", lambda *args, **kwargs: TIMELINE)
    studio = FakeStudio(a_job())
    studio.stop_when = lambda fields: fields.get("progress") == 10 and "failed"  # once the recording is ready

    _, storage = fakes(a_job(), studio=studio)

    assert f"{FOLDER}/job.json" in storage.files


def test_when_studio_is_unreachable_the_job_waits_instead_of_failing(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    studio = FakeStudio(a_job(speaker_labels={"1": "teacher", "2": "translator"}))
    studio.unavailable_when = lambda fields: fields.get("status") == "transcript_ready"

    with pytest.raises(StudioUnavailable):
        fakes(studio.data, storage, studio=studio)

    assert "failed" not in studio.statuses()  # the worker puts it back on the queue


def test_a_missed_progress_report_does_not_stop_the_job(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    studio = FakeStudio(a_job(speaker_labels={"1": "teacher", "2": "translator"}))
    studio.unavailable_when = lambda fields: set(fields) == {"progress"}

    fakes(studio.data, storage, studio=studio)

    assert studio.statuses()[-1] == "transcript_ready"


def test_a_refused_monlam_key_stops_at_once_with_a_plain_reason(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    FakeMonlam.refuse = 401

    studio, _ = fakes(a_job(speaker_labels={"1": "teacher", "2": "translator"}), storage)

    assert studio.reports[-1]["status"] == "failed"
    assert studio.reports[-1]["error"].startswith("Monlam refused this server's key")


def test_out_of_credit_says_so(fakes):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    FakeMonlam.refuse = 402

    studio, _ = fakes(a_job(speaker_labels={"1": "teacher", "2": "translator"}), storage)

    assert "run out of credit" in studio.reports[-1]["error"]


def test_no_speech_at_all_is_explained(fakes, monkeypatch):
    def nothing(*args, **kwargs):
        raise voices.NoSpeech("No speech was found in the recording.")

    monkeypatch.setattr(voices, "find_voices", nothing)

    studio, _ = fakes(a_job())

    assert studio.reports[-1]["error"].startswith("No one could be heard speaking")


def test_progress_never_goes_back_when_garbled_pieces_are_sent_again(fakes, monkeypatch):
    storage = FakeStorage({f"{FOLDER}/1-voices/timeline.json": TIMELINE, f"{FOLDER}/2-clean/clean.flac": b"x"})
    # The first reply for each piece is garbled; the halves come back fine.
    monkeypatch.setattr(
        job_module.pieces, "is_garbled", lambda piece, result: not piece["id"].endswith(("a", "b"))
    )
    monkeypatch.setattr(job_module.Progress, "__init__", _every_report(job_module.Progress.__init__))

    studio, _ = fakes(a_job(speaker_labels={"1": "teacher", "2": "translator"}), storage)

    progress = [r["progress"] for r in studio.reports if set(r) == {"progress"}]
    assert progress == sorted(progress) and progress[-1] >= 90


def _every_report(init):
    def no_throttle(self, *args, **kwargs):
        init(self, *args, **{**kwargs, "every_seconds": 0})

    return no_throttle


# ---------- phase 2: the course ----------

TRANSCRIPT = [
    {"start": 0.0, "end": 15.0, "speaker": "teacher", "text": "ཀ།", "warnings": []},
    {"start": 15.0, "end": 30.0, "speaker": "teacher", "text": "ཁ།", "warnings": []},
    {"start": 30.0, "end": 40.0, "speaker": "no speech", "text": None, "warnings": []},
    {"start": 40.0, "end": 70.0, "speaker": "translator", "text": None, "warnings": []},
    {"start": 70.0, "end": 80.0, "speaker": "teacher", "text": "ག།", "warnings": []},
]
TRANSCRIBED = {"transcript": "3-transcript/transcript.txt", "transcript_json": "3-transcript/transcript.json"}


class FakeLLM:
    refuse = None  # an error to raise instead of answering
    calls = 0

    def __init__(self, api_key, model):
        self.tokens_in = self.tokens_out = 0

    def json(self, prompt, schema, system=""):
        FakeLLM.calls += 1
        if FakeLLM.refuse:
            raise FakeLLM.refuse
        if schema is suggest.OUTLINE_SCHEMA:
            return {"course_title": "སྤྱོད་འཇུག", "sections": [{"title": "ལེའུ།", "lessons": [
                {"title": "ཀ", "first_line": 0, "last_line": 0},
                {"title": "ཁ", "first_line": 1, "last_line": 4},
            ]}]}
        return {"summary": "བསྡུས།", "key_points": ["ཀ"], "mcq": [], "reflection": ["བསམ།"]}


def a_draft():
    def lesson(lesson_id, first, last, start, end, include=True):
        return {"id": lesson_id, "title": "ཀ", "include": include, "first_line": first, "last_line": last,
                "start": start, "end": end, "minutes": round((end - start) / 60, 1), "summary": "",
                "key_points": [], "mcq": [], "reflection": []}

    return {"version": 1, "course_title": "", "sections": [
        {"id": "s1", "title": "ལེའུ།", "lessons": [
            lesson("s1-l1", 0, 0, 0.0, 15.0), lesson("s1-l2", 1, 3, 15.0, 70.0),
        ]},
        {"id": "s2", "title": "ལེའུ།", "lessons": [lesson("s2-l1", 4, 4, 70.0, 80.0, include=False)]},
    ]}


def transcribed_storage(**files):
    return FakeStorage({f"{FOLDER}/3-transcript/transcript.json": TRANSCRIPT, f"{FOLDER}/2-clean/clean.flac": b"x",
                        **files})


@pytest.fixture
def course_fakes(fakes, monkeypatch):
    monkeypatch.setattr(job_module, "LLM", FakeLLM)
    FakeLLM.refuse, FakeLLM.calls = None, 0
    cut = []

    def mp3(sound, rate, dest):
        dest.write_bytes(b"mp3")
        cut.append(dest.name)
        return dest

    monkeypatch.setattr(job_module.lessons, "write_mp3", mp3)

    def run(job, storage=None, env=None, studio=None):
        return fakes(job, storage, {"OPENROUTER_API_KEY": "openrouter-key", **(env or {})}, studio)

    run.cut = cut
    return run


def test_once_the_transcript_is_checked_a_course_is_suggested(course_fakes):
    # Writing the course needs no Monlam: only the AI model.
    studio, storage = course_fakes(a_job(results=TRANSCRIBED), transcribed_storage(), env={"MONLAM_API_KEY": ""})

    assert studio.statuses() == ["suggesting", "course_draft_ready"]
    final = studio.reports[-1]
    assert final["progress"] == 100 and final["results"] == {"course_suggestion": "4-course/suggestion.json"}
    draft = final["course_draft"]
    assert [lesson["id"] for lesson in draft["sections"][0]["lessons"]] == ["s1-l1", "s1-l2"]
    assert draft["sections"][0]["lessons"][1]["reflection"] == ["བསམ།"]
    assert storage.files[f"{FOLDER}/4-course/suggestion.json"] == draft
    assert FakeLLM.calls == 3  # the outline, then each lesson


def test_suggesting_needs_an_openrouter_key(course_fakes):
    studio, _ = course_fakes(a_job(results=TRANSCRIBED), transcribed_storage(), env={"OPENROUTER_API_KEY": ""})
    assert studio.statuses() == ["failed"]
    assert studio.reports[-1]["error"] == "Writing the course isn't set up on this server yet (no OpenRouter key)."


def test_a_refused_key_or_no_credit_is_explained(course_fakes):
    FakeLLM.refuse = LLMRefused(401, "No auth credentials found")
    studio, _ = course_fakes(a_job(results=TRANSCRIBED), transcribed_storage())
    assert studio.reports[-1]["error"].startswith("OpenRouter refused this server's key")

    FakeLLM.refuse = LLMRefused(402, "Insufficient credits")
    studio, _ = course_fakes(a_job(results=TRANSCRIBED), transcribed_storage())
    assert studio.reports[-1]["error"] == (
        "The OpenRouter account has run out of credit. Once it is topped up, try again."
    )


def test_the_model_not_answering_asks_the_author_to_try_again(course_fakes):
    FakeLLM.refuse = LLMError("OpenRouter did not answer after 5 tries")
    studio, _ = course_fakes(a_job(results=TRANSCRIBED), transcribed_storage())
    assert studio.statuses()[-1] == "failed"
    assert "Try again in a few minutes" in studio.reports[-1]["error"]


def test_the_approved_draft_is_cut_into_lessons_and_studio_writes_the_course(course_fakes):
    job = a_job(results=TRANSCRIBED, course_draft=a_draft(), course_approved=True)

    studio, storage = course_fakes(job, transcribed_storage())

    assert studio.statuses() == ["cutting_audio", "writing_course"]
    # The lesson switched off isn't cut; the second lesson has the translator's reply too.
    assert studio.written == {
        "s1-l1": {"tracks": {"bo": "5-lessons/lesson-1-1_bo.mp3"}},
        "s1-l2": {"tracks": {"bo": "5-lessons/lesson-1-2_bo.mp3", "translator": "5-lessons/lesson-1-2_translator.mp3"}},
    }
    for path in ("lesson-1-1_bo.mp3", "lesson-1-2_bo.mp3", "lesson-1-2_translator.mp3"):
        assert storage.files[f"{FOLDER}/5-lessons/{path}"] == b"mp3"
    index = storage.files[f"{FOLDER}/5-lessons/lessons.json"]
    assert index["s1-l2"]["number"] == "1-2" and index["s1-l2"]["pieces"]["translator"][0][2] == "translator"
    assert FakeLLM.calls == 0


def test_cutting_continues_with_the_lessons_not_cut_yet(course_fakes):
    done = {"s1-l1": {"number": "1-1", "tracks": {"bo": "5-lessons/lesson-1-1_bo.mp3"}, "minutes": {}, "pieces": {}}}
    storage = transcribed_storage(**{f"{FOLDER}/5-lessons/lessons.json": done,
                                     f"{FOLDER}/5-lessons/lesson-1-1_bo.mp3": b"mp3"})

    studio, _ = course_fakes(a_job(results=TRANSCRIBED, course_draft=a_draft(), course_approved=True), storage)

    assert sorted(course_fakes.cut) == ["lesson-1-2_bo.mp3", "lesson-1-2_translator.mp3"]
    assert set(studio.written) == {"s1-l1", "s1-l2"}


def test_without_the_cleaned_recording_the_author_is_told(course_fakes):
    storage = FakeStorage({f"{FOLDER}/3-transcript/transcript.json": TRANSCRIPT})
    studio, _ = course_fakes(a_job(results=TRANSCRIBED, course_draft=a_draft(), course_approved=True), storage)
    assert studio.reports[-1]["error"].startswith("The cleaned recording is missing")


def test_starting_over_while_suggesting_removes_what_it_wrote(course_fakes, monkeypatch):
    studio = FakeStudio(a_job(results=TRANSCRIBED))
    # The author starts over just as the outline is done.
    studio.stop_when = lambda fields: (
        "cancelled" if fields.get("progress", 0) >= 30 and "status" not in fields else None
    )

    _, storage = course_fakes(studio.data, transcribed_storage(), studio=studio)

    assert not [key for key in storage.files if "/4-course/" in key]
    assert FakeLLM.calls == 1


def test_a_job_waiting_for_the_author_to_shape_the_course_is_left_alone(course_fakes):
    studio, _ = course_fakes(a_job(status="course_draft_ready", results=TRANSCRIBED, course_draft=a_draft()))
    assert studio.reports == [] and FakeLLM.calls == 0


def test_when_every_lesson_is_cut_it_goes_straight_to_writing(course_fakes):
    tracks = {"s1-l1": {"bo": "5-lessons/lesson-1-1_bo.mp3"}, "s1-l2": {"bo": "5-lessons/lesson-1-2_bo.mp3"}}
    done = {lesson: {"number": "", "tracks": paths, "minutes": {}, "pieces": {}} for lesson, paths in tracks.items()}
    files = {f"{FOLDER}/{path}": b"mp3" for paths in tracks.values() for path in paths.values()}
    storage = transcribed_storage(**{f"{FOLDER}/5-lessons/lessons.json": done}, **files)

    # E.g. a run after Studio said it was still writing the course.
    studio, _ = course_fakes(a_job(results=TRANSCRIBED, course_draft=a_draft(), course_approved=True,
                                   status="writing_course"), storage)

    assert studio.statuses() == ["writing_course"]  # not back to cutting
    assert course_fakes.cut == []
    assert set(studio.written) == {"s1-l1", "s1-l2"}


def test_a_lesson_with_no_sound_of_the_teacher_is_named(course_fakes, monkeypatch):
    real = job_module.lessons.cut_lesson

    def nothing_for_the_first(clean, speech, rate, turns, start, end):
        return {} if start == 0.0 else real(clean, speech, rate, turns, start, end)

    monkeypatch.setattr(job_module.lessons, "cut_lesson", nothing_for_the_first)
    draft = a_draft()
    draft["sections"][0]["lessons"][0]["title"] = "སྐྱབས་འགྲོ།"

    studio, _ = course_fakes(a_job(results=TRANSCRIBED, course_draft=draft, course_approved=True),
                             transcribed_storage())

    assert studio.statuses()[-1] == "failed"
    assert "“སྐྱབས་འགྲོ།”" in studio.reports[-1]["error"]
    assert studio.written is None  # Studio wasn't asked to write a course missing a lesson's audio
