"""
Cutting the teacher's parts into pieces to transcribe, and checking what comes back. Ported from the
pipeline's transcribe.py:

    1. Keep only the turns with the chosen names (the teacher).
    2. Widen each turn to the nearest pause on both sides (up to MAX_PAD s), so the first/last words
       survive a speaker switch that is a few seconds off - but only into silence (or the recording's
       start and end). Next to anyone else's turn, or another teacher turn, the piece stops at the turn's
       edge: no piece takes in another voice, and no second is sent (and paid for) twice.
    3. Split long turns into 10-30 s pieces, cutting only in pauses. With no pause, a piece may grow
       to MAX_PIECE s; only then is it cut at the quietest moment (flagged).
    4. A garbled reply (far too much text, or a phrase stuck repeating - a known Monlam glitch) has its
       piece cut in half at a pause and both halves sent again.
"""

import re
from typing import List

import numpy as np

from .voices import frame_power

SR = 16000
FRAME = 0.02  # seconds per loudness frame
VOICE_DB = 5  # same "is there sound" rule as the voices step
MIN_PAUSE = 0.3  # a pause must be at least this quiet stretch (s)
MAX_PAD = 3.0  # widen turns to the nearest pause, at most this far (s)
MIN_PIECE, TARGET_PIECE, MAX_PIECE = 10.0, 30.0, 45.0
MIN_HALF = 4.0  # never make halves shorter than this (s)
NO_SPEECH = "no speech"  # the name of a silent turn (voices.NO_SPEECH)

# Warning thresholds, in Tibetan syllables per second of audio (normal speech ~3-6).
TOO_FEW_SYL_PER_S = 1.0
TOO_MANY_SYL_PER_S = 9.0


def find_pauses(audio: np.ndarray) -> np.ndarray:
    """(start, end) seconds of every quiet stretch of at least MIN_PAUSE."""
    db = 10 * np.log10(frame_power(audio, int(FRAME * SR)) + 1e-10)
    quiet = db <= np.percentile(db, 10) + VOICE_DB
    edges = np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (ends - starts) * FRAME >= MIN_PAUSE
    return np.stack([starts[keep] * FRAME, ends[keep] * FRAME], axis=1).reshape(-1, 2)


def pauses_between(pauses: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Pauses whose middle lies in [lo, hi]."""
    mids = pauses.mean(axis=1)
    return pauses[(mids >= lo) & (mids <= hi)]


def widen(start: float, end: float, pauses: np.ndarray, earliest: float, latest: float) -> tuple:
    """
    Move start back / end forward to the nearest pause within MAX_PAD seconds, never before ``earliest``
    or after ``latest`` (with no pause in reach, to as far as allowed).
    """
    lo, hi = max(earliest, start - MAX_PAD), min(latest, end + MAX_PAD)
    before = pauses_between(pauses, lo, start)
    new_start = before[-1].mean() if len(before) else lo
    after = pauses_between(pauses, end, hi)
    new_end = after[0].mean() if len(after) else hi
    return float(new_start), float(new_end)


def room(turns: List[dict], idx: int, total: float) -> tuple:
    """How far turn ``idx`` may widen: across silence next to it, but not into any other turn."""
    turn = turns[idx]
    before = turns[idx - 1] if idx > 0 else None
    after = turns[idx + 1] if idx + 1 < len(turns) else None
    earliest = 0.0 if before is None else (before["start"] if before["name"] == NO_SPEECH else turn["start"])
    latest = total if after is None else (after["end"] if after["name"] == NO_SPEECH else turn["end"])
    return earliest, latest


def quietest(audio: np.ndarray, lo: float, hi: float) -> float:
    """The quietest 20 ms moment between ``lo`` and ``hi`` seconds."""
    return lo + float(np.argmin(frame_power(audio[int(lo * SR) : int(hi * SR)], int(FRAME * SR)))) * FRAME


def split(start: float, end: float, pauses: np.ndarray, audio: np.ndarray) -> list:
    """Split [start, end] into pieces cut in pauses. Returns (start, end, forced_cut) tuples."""
    pieces, cur = [], start
    while end - cur > TARGET_PIECE:
        cut, forced = None, False
        for lo, hi in ((MIN_PIECE, TARGET_PIECE), (TARGET_PIECE, MAX_PIECE)):
            options = pauses_between(pauses, cur + lo, min(cur + hi, end - 2))
            if len(options):
                longest = options[np.argmax(options[:, 1] - options[:, 0])]
                cut = float(longest.mean())
                break
        if cut is None:
            if end - cur <= MAX_PIECE:
                break
            # No pause at all: cut at the quietest 20 ms between TARGET_PIECE and MAX_PIECE.
            cut, forced = quietest(audio, cur + TARGET_PIECE, cur + MAX_PIECE), True
        pieces.append((cur, cut, forced))
        cur = cut
    pieces.append((cur, end, False))
    return pieces


def plan_pieces(audio: np.ndarray, pauses: np.ndarray, turns: List[dict], names: set) -> List[dict]:
    total = len(audio) / SR
    plan = []
    for idx, turn in enumerate(turns):
        if turn["name"] not in names:
            continue
        start, end = widen(turn["start"], turn["end"], pauses, *room(turns, idx, total))
        for n, (a, b, forced) in enumerate(split(start, end, pauses, audio)):
            plan.append(
                {
                    "id": f"{len(plan) + 1:04d}",
                    "start": round(float(a), 2),
                    "end": round(float(b), 2),
                    "label": turn["name"],
                    "segment": idx,
                    "segment_edge": bool(n == 0 or b >= end),
                    "forced_cut": bool(forced),
                }
            )
    return plan


def syllables(text: str) -> list:
    return [s for s in re.split(r"[་།༎\s]+", text) if s]


def warnings_for(piece: dict, result: dict) -> list:
    text = (result.get("text") or "").strip()
    seconds = piece["end"] - piece["start"]
    syl = syllables(text)
    rate = len(syl) / seconds if seconds else 0
    found = []
    if not text:
        found.append("empty result")
    elif rate < TOO_FEW_SYL_PER_S:
        found.append(f"too little text ({rate:.1f} syllables/s) - words may be missing")
    elif rate > TOO_MANY_SYL_PER_S:
        found.append(f"too much text ({rate:.1f} syllables/s) - text may be invented")
    for n in (3, 4, 5):  # the same phrase three or more times in a row
        grams = [tuple(syl[i : i + n]) for i in range(len(syl) - n + 1)]
        if any(grams[i] == grams[i + n] == grams[i + 2 * n] for i in range(len(grams) - 2 * n)):
            found.append("a phrase repeats several times")
            break
    if piece["forced_cut"]:
        found.append("cut without a pause - a word may be split")
    return found


def is_garbled(piece: dict, result: dict) -> bool:
    return any(w.startswith(("too much text", "a phrase repeats")) for w in warnings_for(piece, result))


def halve(piece: dict, pauses: np.ndarray, audio: np.ndarray) -> list:
    """Cut a piece in two at the longest pause near its middle (or the quietest moment)."""
    length = piece["end"] - piece["start"]
    lo, hi = piece["start"] + 0.35 * length, piece["start"] + 0.65 * length
    options = pauses_between(pauses, lo, hi)
    forced = not len(options)
    cut = quietest(audio, lo, hi) if forced else float(options[np.argmax(options[:, 1] - options[:, 0])].mean())
    cut = round(cut, 2)
    return [
        {**piece, "id": piece["id"] + "a", "end": cut, "forced_cut": piece["forced_cut"] or forced},
        {**piece, "id": piece["id"] + "b", "start": cut, "forced_cut": piece["forced_cut"] or forced},
    ]
