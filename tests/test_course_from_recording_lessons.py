"""Cutting each lesson's audio (course_from_recording/lessons.py)."""

import numpy as np
import soundfile as sf

from studio_tool_service import audio
from studio_tool_service.course_from_recording import lessons

SPEECH_SR, CLEAN_SR = 16000, 44100


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def test_neighbouring_lines_of_one_voice_are_one_turn():
    lines = [
        {"start": 0.0, "end": 10.0, "speaker": "teacher"},
        {"start": 10.0, "end": 20.0, "speaker": "teacher"},
        {"start": 20.0, "end": 30.0, "speaker": "translator"},
        {"start": 30.0, "end": 35.0, "speaker": "no speech"},
        {"start": 35.0, "end": 50.0, "speaker": "teacher"},
    ]
    assert lessons.turns_from_lines(lines) == [
        turn(0.0, 20.0, "teacher"), turn(20.0, 30.0, "translator"), turn(30.0, 35.0, "no speech"),
        turn(35.0, 50.0, "teacher"),
    ]


def test_a_switch_moves_to_the_longest_pause_nearby():
    pauses = np.array([[9.0, 9.3], [10.5, 11.5], [13.0, 14.0]])
    assert lessons.snap(10.0, pauses) == 11.0  # the 1 s pause, not the closer short one
    assert lessons.snap(20.0, pauses) == 20.0  # nothing within 2 s: left as it is
    assert lessons.snap(5.0, np.zeros((0, 2))) == 5.0


def test_each_voice_goes_to_its_track():
    turns = [
        turn(0.0, 30.0, "chanting"), turn(30.0, 60.0, "teacher"), turn(60.0, 90.0, "translator"),
        turn(90.0, 95.0, "other"), turn(95.0, 100.0, "no speech"), turn(100.0, 130.0, "teacher"),
    ]
    tracks = lessons.lesson_pieces(turns, 0.0, 120.0, np.zeros((0, 2)))
    # Chanting and the teacher are next to each other: one piece. The lesson ends at 120 s.
    assert tracks["bo"] == [
        {"start": 0.0, "end": 60.0, "voices": "chanting+teacher"},
        {"start": 100.0, "end": 120.0, "voices": "teacher"},
    ]
    assert tracks["translator"] == [{"start": 60.0, "end": 90.0, "voices": "translator"}]


def test_slivers_are_left_out_and_tracks_with_nothing_dropped():
    turns = [turn(0.0, 30.0, "teacher"), turn(30.0, 31.2, "translator"), turn(31.2, 60.0, "teacher")]
    tracks = lessons.lesson_pieces(turns, 0.5, 59.5, np.zeros((0, 2)))
    assert set(tracks) == {"bo"}  # 1.2 s of translator is too short
    # A turn only just inside the lesson (less than 1 s of it) doesn't count either.
    assert lessons.lesson_pieces(turns, 59.2, 90.0, np.zeros((0, 2))) == {}


def test_pieces_are_joined_with_a_short_silence_and_faded():
    sound = np.ones(10 * CLEAN_SR, dtype=np.float32)
    track = [{"start": 1.0, "end": 3.0, "voices": "teacher"}, {"start": 5.0, "end": 6.0, "voices": "teacher"}]

    joined = lessons.join(sound, CLEAN_SR, 0.0, track)

    assert len(joined) == int(2.0 * CLEAN_SR) + int(lessons.GAP * CLEAN_SR) + int(1.0 * CLEAN_SR)
    assert joined[0] == 0.0 and joined[CLEAN_SR] == 1.0  # faded in, then full
    gap_start = int(2.0 * CLEAN_SR)
    assert not joined[gap_start: gap_start + int(lessons.GAP * CLEAN_SR)].any()


def speaking(seconds, freq, rate):
    """Phrases of a tone with short breaks, like speech with pauses between phrases."""
    t = np.arange(int(1.7 * rate)) / rate
    phrase = np.concatenate([0.3 * np.sin(2 * np.pi * freq * t), np.zeros(int(0.3 * rate))])
    return np.tile(phrase, int(seconds / 2)).astype(np.float32)


def recording(rate):
    """30 s teacher, 10 s silence, 30 s translator."""
    return np.concatenate([speaking(30, 220, rate), np.zeros(10 * rate, np.float32), speaking(30, 330, rate)]) + 0.001


def test_cuts_a_lesson_from_real_files(tmp_path):
    speech, clean = tmp_path / "speech.wav", tmp_path / "clean.flac"
    sf.write(str(speech), recording(SPEECH_SR), SPEECH_SR, subtype="PCM_16")
    sf.write(str(clean), recording(CLEAN_SR), CLEAN_SR, subtype="PCM_16")
    turns = [turn(0.0, 30.0, "teacher"), turn(30.0, 40.0, "no speech"), turn(40.0, 70.0, "translator")]

    tracks = lessons.cut_lesson(clean, speech, CLEAN_SR, turns, 0.0, 70.0)

    assert set(tracks) == {"bo", "translator"}
    for name, expected in (("bo", 30.0), ("translator", 30.0)):
        seconds = len(tracks[name]["sound"]) / CLEAN_SR
        assert abs(seconds - expected) < 2 * lessons.SNAP, (name, seconds)

    mp3 = lessons.write_mp3(tracks["bo"]["sound"], CLEAN_SR, tmp_path / "lesson-1-1_bo.mp3")
    assert mp3.exists() and not (tmp_path / "lesson-1-1_bo.wav").exists()
    assert abs(audio.duration(mp3) - len(tracks["bo"]["sound"]) / CLEAN_SR) < 0.2


def test_a_lesson_with_only_short_bits_of_the_teacher_still_gets_its_tibetan_track(tmp_path):
    speech, clean = tmp_path / "speech.wav", tmp_path / "clean.flac"
    sf.write(str(speech), recording(SPEECH_SR), SPEECH_SR, subtype="PCM_16")
    sf.write(str(clean), recording(CLEAN_SR), CLEAN_SR, subtype="PCM_16")
    turns = [turn(0.0, 30.0, "teacher"), turn(30.0, 40.0, "no speech"), turn(40.0, 70.0, "translator")]
    # The lesson has only the last 0.8 s of the teacher's turn: too little to count, normally.
    assert "bo" not in lessons.lesson_pieces(turns, 29.2, 70.0, np.zeros((0, 2)))

    tracks = lessons.cut_lesson(clean, speech, CLEAN_SR, turns, 29.2, 70.0)

    assert set(tracks) == {"bo", "translator"}
    assert len(tracks["bo"]["sound"]) > 0
