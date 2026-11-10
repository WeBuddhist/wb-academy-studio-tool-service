"""
Phase 2, step 2: cut each lesson's audio, as the pipeline's cut_unit_audio.py does.

For one lesson (a stretch of the recording), from the cleaned recording:

    - the teacher and chanting go to the Tibetan track ("bo"), the translator to the translator track;
      anyone else and the silences are left out;
    - each switch between voices is moved to the longest real pause within SNAP seconds, so no word is cut
      and none of the other language leaks in;
    - pieces closer than MERGE_GAP are joined into one, and pieces shorter than MIN_PIECE dropped (unless
      the teacher has nothing longer in the lesson: every lesson has its Tibetan track);
    - the rest are put together with GAP seconds of silence between them and very short fades.

Who speaks when comes from the transcript (3-transcript/transcript.json), which has a line for every
stretch of the recording. The pauses are found in the 16 kHz working copy of the original, over the
lesson's own stretch only (so a quiet lesson isn't judged against a loud one).
"""

import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from .. import audio
from . import pieces

SNAP = 2.0  # move a switch to the longest pause within this many seconds
GAP = 0.6  # silence put between two pieces (s)
FADE = 0.02  # fade in and out of each piece (s)
MIN_PIECE = 1.5  # shorter pieces are left out (s)
MERGE_GAP = 0.5  # pieces closer than this are one piece (s)
MIN_OVERLAP = 1.0  # a turn must take up at least this much of the lesson to count (s)
BITRATE = "64k"  # mono speech: about 0.5 MB a minute

# Which track each voice goes to; voices not listed ("other", "no speech") are left out.
TRACKS = {"teacher": "bo", "chanting": "bo", "translator": "translator"}


def turns_from_lines(lines: List[dict]) -> List[dict]:
    """The transcript's lines as turns: neighbouring lines of the same voice joined."""
    turns = []
    for line in sorted(lines, key=lambda item: item["start"]):
        speaker = line.get("speaker")
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end"] = max(turns[-1]["end"], line["end"])
        else:
            turns.append({"start": float(line["start"]), "end": float(line["end"]), "speaker": speaker})
    return turns


def snap(moment: float, pauses: np.ndarray) -> float:
    """The middle of the longest pause within SNAP seconds of ``moment``, or ``moment`` if there is none."""
    if not len(pauses):
        return moment
    middles = pauses.mean(axis=1)
    near = pauses[np.abs(middles - moment) <= SNAP]
    if not len(near):
        return moment
    longest = near[np.argmax(near[:, 1] - near[:, 0])]
    return float(longest.mean())


def lesson_pieces(
    turns: List[dict], start: float, end: float, pauses: np.ndarray, min_overlap: float = MIN_OVERLAP,
    min_piece: float = MIN_PIECE,
) -> Dict[str, List[dict]]:
    """Each track's pieces of the lesson ``start``-``end``: [{start, end, voices}] in seconds of the recording."""
    tracks: Dict[str, List[dict]] = {}
    for turn in turns:
        track = TRACKS.get(turn["speaker"])
        lo, hi = max(turn["start"], start), min(turn["end"], end)
        if track is None or hi - lo < min_overlap or hi <= lo:
            continue
        lo, hi = snap(lo, pauses), snap(hi, pauses)
        if hi > lo:
            tracks.setdefault(track, []).append({"start": lo, "end": hi, "voices": turn["speaker"]})

    joined = {}
    for track, found in tracks.items():
        merged: List[dict] = []
        for piece in sorted(found, key=lambda item: item["start"]):
            if merged and piece["start"] - merged[-1]["end"] < MERGE_GAP:
                merged[-1]["end"] = max(merged[-1]["end"], piece["end"])
                if piece["voices"] not in merged[-1]["voices"].split("+"):
                    merged[-1]["voices"] += "+" + piece["voices"]
            else:
                merged.append(dict(piece))
        kept = [piece for piece in merged if piece["end"] - piece["start"] >= min_piece]
        if kept:
            joined[track] = kept
    return joined


def join(sound: np.ndarray, rate: int, offset: float, track: List[dict]) -> np.ndarray:
    """One track's pieces of ``sound`` (which starts at ``offset`` s in the recording), with silence between."""
    fade = int(FADE * rate)
    gap = np.zeros(int(GAP * rate), dtype=np.float32)
    parts = []
    for number, piece in enumerate(track):
        clip = sound[max(0, int((piece["start"] - offset) * rate)): max(0, int((piece["end"] - offset) * rate))].copy()
        if len(clip) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        if number:
            parts.append(gap)
        parts.append(clip.astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def cut_lesson(clean: Path, speech: Path, clean_rate: int, turns: List[dict], start: float, end: float) -> dict:
    """
    The lesson's tracks: {track: {"sound": samples, "pieces": [...]}}, with ``clean`` the cleaned recording
    (at ``clean_rate``) and ``speech`` the 16 kHz working copy the pauses are found in.
    """
    lo, hi = max(0.0, start - SNAP - 1), end + SNAP + 1
    window = audio.read(speech, lo, hi)
    pauses = pieces.find_pauses(window) + lo if len(window) else np.zeros((0, 2))
    found = lesson_pieces(turns, start, end, pauses)
    if "bo" not in found:
        # Only short bits of the teacher (or chanting) in this lesson: keep them all rather than none, since
        # the course needs every lesson's Tibetan track.
        short = lesson_pieces(turns, start, end, pauses, min_overlap=0.0, min_piece=0.0)
        if "bo" in short:
            found["bo"] = short["bo"]
    if not found:
        return {}
    sound = audio.read(clean, lo, hi)
    return {track: {"sound": join(sound, clean_rate, lo, track_pieces), "pieces": track_pieces}
            for track, track_pieces in found.items()}


def write_mp3(sound: np.ndarray, rate: int, dest: Path) -> Path:
    """Mono MP3 at BITRATE, through a temporary WAV next to ``dest``."""
    wav = dest.with_suffix(".wav")
    sf.write(str(wav), sound, rate, subtype="PCM_16")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-c:a", "libmp3lame",
             "-b:a", BITRATE, str(dest)],
            check=True,
        )
    finally:
        wav.unlink(missing_ok=True)
    return dest
