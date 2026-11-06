"""The step logic of "course from a recording", on generated audio (tones and silence)."""

import numpy as np
import pytest
import soundfile as sf

from studio_tool_service.course_from_recording import clean, pieces, transcript, voices
from studio_tool_service.course_from_recording import studio as studio_module
from studio_tool_service.course_from_recording.monlam import Transcriber

SR = 16000


def tone(seconds, freq=220.0, level=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def hiss(seconds, level=0.002, sr=SR, seed=0):
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * sr))).astype(np.float32)


def speaking(seconds, sr=SR):
    """Like speech: 1.7 s phrases with 0.3 s breaks between them (as quiet as the room)."""
    phrase, gap = tone(1.7, sr=sr), np.zeros(int(0.3 * sr), np.float32)
    out = np.tile(np.concatenate([phrase, gap]), int(np.ceil(seconds / 2)))[: int(seconds * sr)]
    return out + hiss(seconds, sr=sr)


def talk(parts, sr=SR):
    """[(seconds, speaking?), ...] -> audio: phrases while speaking, faint hiss otherwise."""
    return np.concatenate([speaking(s, sr=sr) if on else hiss(s, sr=sr, seed=i) for i, (s, on) in enumerate(parts)])


# ---------- voices ----------


def test_one_speaker_is_the_teacher_apart_from_long_silences():
    audio = talk([(30, True), (8, False), (40, True), (2, False), (20, True)])

    turns = voices.speech_turns(audio)

    assert [t["name"] for t in turns] == ["teacher", "no speech", "teacher"]
    assert 28 < turns[1]["start"] < 32 and 36 < turns[1]["end"] < 40  # the 8 s silence
    assert turns[-1]["end"] > 85  # the 2 s pause stays with the teacher


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def voice_cloud(center, count, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return unit(center + spread * rng.standard_normal((count, len(center))))


def test_finds_how_many_voices_there_are_biggest_first():
    rng = np.random.default_rng(1)
    centers = unit(rng.standard_normal((3, 192)))
    emb = np.concatenate(
        [voice_cloud(centers[0], 900), voice_cloud(centers[1], 600, seed=2), voice_cloud(centers[2], 300, seed=3)]
    )

    found = voices.group_voices(emb)

    assert len(found) == 3
    assert [int(np.argmax(found[i] @ centers.T)) for i in range(3)] == [0, 1, 2]


def test_a_voice_heard_for_under_a_minute_is_merged_away():
    rng = np.random.default_rng(4)
    centers = unit(rng.standard_normal((3, 192)))
    # 40 windows * 0.75 s = 30 s for the third voice.
    emb = np.concatenate(
        [voice_cloud(centers[0], 900), voice_cloud(centers[1], 600, seed=5), voice_cloud(centers[2], 40, seed=6)]
    )

    assert len(voices.group_voices(emb)) == 2


def test_the_authors_tags_name_the_turns():
    timeline = [
        {"start": 0, "end": 60, "group": "1"},
        {"start": 60, "end": 70, "group": None},
        {"start": 70, "end": 100, "group": "2"},
        {"start": 100, "end": 110, "group": "3"},
    ]

    named = voices.name_turns(timeline, {"1": "teacher", "2": "translator"})

    assert [t["name"] for t in named] == ["teacher", "no speech", "translator", "other"]


def test_each_voice_gets_its_minutes_and_samples_from_its_longest_turns():
    timeline = [
        {"start": 0, "end": 30, "group": "1"},
        {"start": 30, "end": 40, "group": "2"},
        {"start": 40, "end": 130, "group": "1"},
        {"start": 130, "end": 140, "group": None},
    ]

    groups = voices.voice_groups(timeline)

    assert [(g["group"], g["minutes"]) for g in groups] == [("1", 2.0), ("2", 0.2)]
    # Three clips for voice 1: two spread over its 90 s turn, one from its 30 s turn.
    assert voices.sample_spans(groups[0]["longest"], 3, 20.0) == [(5.0, 20.0), (60.0, 20.0), (90.0, 20.0)]
    # Voice 2 only speaks for 10 s: that whole turn is its one clip.
    assert voices.sample_spans(groups[1]["longest"], 3, 20.0) == [(30.0, 10.0)]


# ---------- pieces ----------


def test_long_teacher_turns_are_cut_in_pauses_into_short_pieces():
    parts = []
    for _ in range(8):
        parts += [(12, True), (0.6, False)]
    audio = talk(parts)
    pauses = pieces.find_pauses(audio)
    turns = [{"start": 0.0, "end": len(audio) / SR, "name": "teacher"}]

    plan = pieces.plan_pieces(audio, pauses, turns, {"teacher"})

    assert len(plan) > 1
    assert all(p["end"] - p["start"] <= pieces.MAX_PIECE for p in plan)
    assert not any(p["forced_cut"] for p in plan)
    assert plan[0]["start"] == 0.0 and plan[-1]["end"] == round(len(audio) / SR, 2)


def test_only_the_teacher_is_sent():
    audio = talk([(20, True), (20, True)])
    turns = [{"start": 0.0, "end": 20.0, "name": "translator"}, {"start": 20.0, "end": 40.0, "name": "teacher"}]

    plan = pieces.plan_pieces(audio, pieces.find_pauses(audio), turns, {"teacher"})

    assert {p["label"] for p in plan} == {"teacher"} and plan[0]["start"] == 20.0


def test_a_teacher_piece_never_runs_into_another_voice_but_may_widen_into_silence():
    # Teacher 0-20, translator 20-35, silence 35-45, teacher 45-60, with short pauses everywhere.
    audio = talk([(2, True), (0.5, False)] * 24)
    turns = [
        {"start": 0.0, "end": 20.0, "name": "teacher"},
        {"start": 20.0, "end": 35.0, "name": "translator"},
        {"start": 35.0, "end": 45.0, "name": "no speech"},
        {"start": 45.0, "end": 60.0, "name": "teacher"},
    ]

    plan = pieces.plan_pieces(audio, pieces.find_pauses(audio), turns, {"teacher"})

    first, second = [p for p in plan if p["segment"] == 0], [p for p in plan if p["segment"] == 3]
    assert first[0]["start"] == 0.0 and first[-1]["end"] == 20.0  # stops at the translator
    assert 42.0 <= second[0]["start"] < 45.0  # widened back into the silence, to a pause
    assert second[-1]["end"] == 60.0


def test_neighbouring_teacher_turns_are_not_sent_twice():
    audio = talk([(2, True), (0.5, False)] * 16)
    turns = [{"start": 0.0, "end": 20.0, "name": "teacher"}, {"start": 20.0, "end": 40.0, "name": "teacher"}]

    plan = pieces.plan_pieces(audio, pieces.find_pauses(audio), turns, {"teacher"})

    spans = sorted((p["start"], p["end"]) for p in plan)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))


def test_a_phrase_stuck_repeating_is_garbled():
    piece = {"start": 0.0, "end": 20.0, "forced_cut": False}
    stuck = "བཀྲ་ཤིས་བདེ་ལེགས་" * 6
    assert pieces.is_garbled(piece, {"text": stuck})
    assert not pieces.is_garbled(piece, {"text": "ང་ཚོས་དེ་རིང་སྤྱོད་འཇུག་གི་ལེའུ་གཉིས་པ་བཤད་ཀྱི་ཡིན།"})


# ---------- transcript ----------


def test_the_particle_follows_the_previous_syllable():
    assert transcript.rule_fix("ཆོས་གི་") == "ཆོས་ཀྱི་"
    assert transcript.rule_fix("བླ་མ་ཀྱི་") == "བླ་མ་གི་"
    assert transcript.rule_fix("ལམ་གི་") == "ལམ་གྱི་"
    assert transcript.rule_fix("སྤྱང་ཀི་") == "སྤྱང་ཀི་"  # wolf, not the particle
    assert transcript.rule_fix("མྱི་") == "མི་"
    assert transcript.rule_fix("ཡིན། །། ། དེ") == "ཡིན། དེ"


def test_lines_are_in_time_order_with_the_untranscribed_turns():
    plan = [
        {
            "id": "0001",
            "start": 10.0,
            "end": 30.0,
            "label": "teacher",
            "segment": 1,
            "segment_edge": True,
            "forced_cut": False,
        }
    ]
    results = {
        "0001": {
            "result": {
                "text": "ཀ་ཁ",
                "segments": [{"start": 0.5, "end": 9.0, "text": " ཀ "}, {"start": 9.0, "end": 19.5, "text": "ཁ"}],
            }
        }
    }
    turns = [{"start": 0.0, "end": 10.0, "name": "translator"}, {"start": 10.0, "end": 30.0, "name": "teacher"}]

    lines = transcript.build_lines(plan, results, turns)

    assert [(line["start"], line["speaker"], line["text"]) for line in lines] == [
        (0.0, "translator", None),
        (10.5, "teacher", "ཀ"),
        (19.0, "teacher", "ཁ"),
    ]
    assert "(translator - not transcribed)" in transcript.as_text(lines, "# heading")


# ---------- clean ----------


def test_static_is_learned_from_silences_and_turned_down(tmp_path):
    sr = clean.SR
    noisy = np.concatenate(
        [hiss(10, level=0.02, sr=sr), tone(10, level=0.3, sr=sr) + hiss(10, level=0.02, sr=sr, seed=1)]
    )
    wav = tmp_path / "work.wav"
    sf.write(str(wav), noisy, sr, subtype="PCM_16")
    turns = [{"start": 0.0, "end": 10.0, "name": "no speech"}, {"start": 10.0, "end": 20.0, "name": "teacher"}]

    noise = clean.noise_power(clean.noise_sample(wav, clean.quiet_spans(turns, np.zeros((0, 2)))))
    out = clean.clean_file(wav, tmp_path / "clean.flac", noise)

    cleaned, out_sr = sf.read(str(out), dtype="float32")
    assert out_sr == sr and len(cleaned) == len(noisy)
    silence_before, silence_after = np.std(noisy[sr : 9 * sr]), np.std(cleaned[sr : 9 * sr])
    assert silence_after < silence_before / 3  # static turned down
    assert np.std(cleaned[11 * sr : 19 * sr]) > 0.8 * np.std(noisy[11 * sr : 19 * sr])  # voice kept


def test_without_long_silences_the_pauses_are_used():
    spans = clean.quiet_spans(
        [{"start": 0.0, "end": 100.0, "name": "teacher"}], np.array([[5.0, 5.5], [20.0, 22.0], [40.0, 41.5]])
    )
    assert spans == [(20.1, 21.9), (40.1, 41.4)]


def test_a_short_recording_keeps_its_brief_voices():
    rng = np.random.default_rng(7)
    centers = unit(rng.standard_normal((2, 192)))
    # 70 + 60 windows = about 50 s and 45 s: each under a minute, but half of a short recording.
    emb = np.concatenate([voice_cloud(centers[0], 70), voice_cloud(centers[1], 60, seed=8)])

    assert len(voices.group_voices(emb)) == 2


def test_without_any_pause_the_short_ones_are_used():
    spans = clean.quiet_spans([{"start": 0.0, "end": 100.0, "name": "teacher"}], np.array([[5.0, 5.4], [20.0, 20.5]]))
    assert spans == [(20.0, 20.5), (5.0, 5.4)]


def test_with_no_quiet_at_all_the_sound_is_kept(tmp_path):
    sr = clean.SR
    wav = tmp_path / "work.wav"
    sf.write(str(wav), tone(3, sr=sr), sr, subtype="PCM_16")

    out = clean.clean_file(wav, tmp_path / "clean.flac", None)

    assert len(sf.read(str(out))[0]) == 3 * sr


def test_the_waveform_is_loud_where_someone_speaks_and_near_zero_in_silence():
    t = np.arange(10 * SR) / SR
    speech = 0.3 * np.sin(2 * np.pi * 220 * t)
    recording = np.concatenate([speech, np.zeros(10 * SR), speech / 6]).astype(np.float32)

    wave = voices.waveform(recording, 30)

    loud, silent, quiet = wave["levels"][:10], wave["levels"][10:20], wave["levels"][20:]
    assert wave["seconds_per_level"] == 1.0
    assert min(loud) > 0.95 and max(silent) == 0
    assert 0.2 < min(quiet) < 0.4  # a sixth as loud, but still visible


def test_when_the_run_stops_the_pieces_not_yet_sent_are_dropped():
    class Stop(Exception):
        pass

    class Api:
        sent = []

        def submit(self, wav, key):
            Api.sent.append(key)
            return key

        def wait(self, job_id):
            return {"status": "COMPLETED", "result": {"text": "x"}}

    class Store:
        def read_json(self, key):
            return None

        def write_json(self, key, data):
            pass

    def stop(done, total, sent):
        if done:
            raise Stop

    pieces_ = [{"id": f"{n:03d}", "start": float(n), "end": n + 1.0} for n in range(30)]
    transcriber = Transcriber(Api(), Store(), "results", np.zeros(31 * SR, np.float32), "job")

    with pytest.raises(Stop):
        transcriber.send_all(pieces_, {}, stop)

    assert len(Api.sent) < 10  # the first reply stops it; at most the few already started finish


# ---------- Studio client ----------


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body or {}

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, *answers):
        self.answers, self.headers, self.calls = list(answers), {}, 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(studio_module.time, "sleep", lambda seconds: None)


def test_studio_is_asked_again_while_it_restarts(no_wait):
    import requests

    session = FakeSession(requests.ConnectionError("refused"), FakeResponse(502), FakeResponse(200, {"id": "j"}))

    assert studio_module.Studio("http://cms", "t", session).job("j") == {"id": "j"}
    assert session.calls == 3


def test_studio_down_for_good_means_try_the_job_later(no_wait):
    session = FakeSession(*[FakeResponse(503)] * 4)

    with pytest.raises(studio_module.StudioUnavailable):
        studio_module.Studio("http://cms", "t", session).report("j", status="cleaning")


def test_a_refused_report_stops_the_job_without_retrying(no_wait):
    session = FakeSession(FakeResponse(409, {"status": "cancelled"}))

    with pytest.raises(studio_module.JobStopped) as stopped:
        studio_module.Studio("http://cms", "t", session).report("j", progress=5)
    assert stopped.value.status == "cancelled" and session.calls == 1


def test_a_progress_report_is_tried_once_and_skipped_if_studio_is_busy(no_wait):
    session = FakeSession(FakeResponse(502), FakeResponse(200))
    progress = studio_module.Progress(studio_module.Studio("http://cms", "t", session), "j", every_seconds=0)

    progress(0.2)  # skipped, not raised
    progress(0.4)

    assert session.calls == 2


# ---------- Monlam request keys ----------


class KeyMonlam:
    def __init__(self, fail_first=0):
        self.keys, self.fail_first = [], fail_first

    def submit(self, wav, key):
        self.keys.append(key)
        return key

    def wait(self, job_id):
        if len(self.keys) <= self.fail_first:
            return {"status": "FAILED", "error": "busy"}
        return {"status": "COMPLETED", "result": {"text": "x"}}


class MemoryStore:
    def __init__(self):
        self.files = {}

    def read_json(self, key):
        return self.files.get(key)

    def write_json(self, key, data):
        self.files[key] = data


def test_a_piece_is_asked_for_with_the_same_key_in_a_later_run(monkeypatch):
    monkeypatch.setattr("studio_tool_service.course_from_recording.monlam.time.sleep", lambda seconds: None)
    piece = {"id": "0001", "start": 0.0, "end": 2.0}
    audio_ = np.zeros(3 * SR, np.float32)
    first, second = KeyMonlam(), KeyMonlam()

    Transcriber(first, MemoryStore(), "r", audio_, "job").transcribe(piece)
    Transcriber(second, MemoryStore(), "r", audio_, "job").transcribe(piece)

    # A run cut off while Monlam worked on the piece asks again with the same key: Monlam doesn't bill twice.
    assert first.keys == second.keys == ["job-0001-0.00-2.00-try1"]


def test_after_a_failure_the_next_request_gets_a_new_key_even_in_a_later_run(monkeypatch):
    monkeypatch.setattr("studio_tool_service.course_from_recording.monlam.time.sleep", lambda seconds: None)
    piece = {"id": "0001", "start": 0.0, "end": 2.0}
    audio_, store = np.zeros(3 * SR, np.float32), MemoryStore()

    Transcriber(KeyMonlam(fail_first=1), store, "r", audio_, "job").transcribe(piece)
    later = KeyMonlam()
    Transcriber(later, store, "r", audio_, "job").transcribe(piece)

    assert later.keys == ["job-0001-0.00-2.00-try2"]  # try1 failed: Monlam would keep refusing that key


# ---------- noise floor and frame power ----------


def test_digital_silence_as_the_noise_sample_does_not_break_the_cleaned_audio(tmp_path):
    sr = clean.SR
    noise = clean.noise_power(np.zeros(sr, np.float32))
    wav = tmp_path / "work.wav"
    sf.write(str(wav), np.concatenate([np.zeros(sr), tone(2, sr=sr)]).astype(np.float32), sr)

    out = clean.clean_file(wav, tmp_path / "clean.flac", noise)

    cleaned, _ = sf.read(str(out))
    assert np.isfinite(noise).all() and np.isfinite(cleaned).all() and np.abs(cleaned).max() > 0.05


def test_frame_power_matches_the_plain_calculation(monkeypatch):
    monkeypatch.setattr(voices, "POWER_CHUNK", 7)  # several chunks, the last one partial
    audio_ = np.random.default_rng(0).normal(size=16000 * 3 + 123).astype(np.float32)

    frames = audio_[: len(audio_) // 800 * 800].reshape(-1, 800)
    assert np.allclose(voices.frame_power(audio_, 800), np.mean(frames.astype(np.float64) ** 2, axis=1))
