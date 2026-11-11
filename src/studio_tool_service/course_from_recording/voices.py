"""
Who speaks when. Ported from the pipeline's detect_speakers.py; the windows, pause rule, smoothing and
turn building are unchanged. Two differences: there are no example times, and the number of voices is
not fixed but found from the recording (the author then says who each voice is).

    1. Cut the 16 kHz audio into 1.5 s windows (0.75 s apart) and mark the ones with sound.
    2. Turn every window with sound into a voice fingerprint (SpeechBrain ECAPA speaker embedding).
    3. Group the fingerprints: try 2 to MAX_VOICES groups and keep the count whose groups are most
       clearly apart (silhouette score); voices heard for less than MIN_VOICE_MINUTES in total (or
       MIN_VOICE_SHARE of the speech, in a short recording) are merged into the nearest bigger one, so
       the author isn't asked about a cough or a phone.
    4. Smooth out flickers and build turns; long stretches without sound become "no speech".

With one speaker only steps 1 and 4 run (``speech_turns``): cleaning still needs the quiet stretches.
"""

from typing import Callable, List

import numpy as np

SR = 16000
WIN = 1.5  # seconds per fingerprint window
HOP = 0.75  # step between windows
SMOOTH = 6  # majority vote over +/- this many windows (~9 s)
MIN_TURN = 4.0  # turns shorter than this (s) are merged into a neighbour
LONG_GAP = 5.0  # stretches without sound this long (s) become their own "no speech" turn
NO_SPEECH = "no speech"
VOICE_DB = 5  # a 50 ms frame counts as sound when this many dB above the noise floor
MAX_VOICES = 6
MIN_VOICE_MINUTES = 1.0
MIN_VOICE_SHARE = 0.1  # ... or this share of all speech, if smaller: in a short recording every voice is brief
SILHOUETTE_SAMPLE = 3000
BATCH = 64
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WAVE_LEVELS = 1000  # loudness values saved for the waveform on the tag voices screen


class NoSpeech(ValueError):
    """Nothing in the recording sounds like speech."""


POWER_CHUNK = 1 << 16  # frames squared at a time


def frame_power(audio: np.ndarray, frame: int) -> np.ndarray:
    """
    The mean power of every ``frame`` samples, a chunk of frames at a time, so no squared copy of the whole
    recording is made (~1 GB for four hours at 16 kHz).
    """
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame)
    power = np.empty(len(frames))
    for i in range(0, len(frames), POWER_CHUNK):
        chunk = frames[i : i + POWER_CHUNK]
        power[i : i + POWER_CHUNK] = np.einsum("ij,ij->i", chunk, chunk) / frame
    return power


def window_starts(n_samples: int) -> np.ndarray:
    win, hop = int(WIN * SR), int(HOP * SR)
    return np.arange(0, max(0, n_samples - win + 1), hop)


def speech_mask(audio: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """True for windows where at least 40% of 50 ms frames are clearly above the noise floor."""
    frame = int(0.05 * SR)
    frame_db = 10 * np.log10(frame_power(audio, frame) + 1e-10)
    voiced = frame_db > np.percentile(frame_db, 10) + VOICE_DB
    per_win = int(WIN / 0.05)
    return np.array([voiced[f : f + per_win].mean() >= 0.4 for f in (starts // frame)], dtype=bool)


def fingerprints(
    audio: np.ndarray,
    starts: np.ndarray,
    speech: np.ndarray,
    model_dir: str,
    threads: int,
    on_progress: Callable[[float], None] = lambda done: None,
) -> np.ndarray:
    """Normalised embeddings of the windows with sound (one row per True in ``speech``)."""
    import torch  # heavy: imported only when the voices are really fingerprinted
    from speechbrain.inference.speaker import EncoderClassifier

    torch.set_num_threads(threads)
    model = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=model_dir, run_opts={"device": "cpu"})
    win = int(WIN * SR)
    picked = starts[speech]
    out = []
    with torch.no_grad():
        for i in range(0, len(picked), BATCH):
            batch = np.stack([audio[s : s + win] for s in picked[i : i + BATCH]])
            out.append(model.encode_batch(torch.from_numpy(batch)).squeeze(1).cpu().numpy())
            on_progress(min(i + BATCH, len(picked)) / max(1, len(picked)))
    emb = np.concatenate(out) if out else np.zeros((0, 192), np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-10)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-10)


def group_voices(emb: np.ndarray) -> np.ndarray:
    """
    Centroids of the voices, biggest first. The count is the one (2..MAX_VOICES) whose groups are
    most clearly apart; voices heard for less than MIN_VOICE_MINUTES are dropped, so their windows
    go to the nearest remaining voice.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(emb) < 2 * MAX_VOICES:
        return unit(emb.mean(axis=0, keepdims=True))
    best_score, best = -2.0, None
    for k in range(2, MAX_VOICES + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(
            emb, km.labels_, metric="cosine", sample_size=min(SILHOUETTE_SAMPLE, len(emb)), random_state=0
        )
        if score > best_score:
            best_score, best = score, km
    centroids = unit(best.cluster_centers_)
    smallest = min(MIN_VOICE_MINUTES, MIN_VOICE_SHARE * len(emb) * HOP / 60)
    for _ in range(len(centroids)):
        counts = np.bincount(np.argmax(emb @ centroids.T, axis=1), minlength=len(centroids))
        minutes = counts * HOP / 60
        if len(centroids) == 1 or minutes.min() >= smallest:
            break
        centroids = np.delete(centroids, int(np.argmin(minutes)), axis=0)
    counts = np.bincount(np.argmax(emb @ centroids.T, axis=1), minlength=len(centroids))
    return centroids[np.argsort(-counts)]


def smooth(labels: np.ndarray) -> np.ndarray:
    """Majority vote over neighbouring windows; -1 (pause) does not vote."""
    out = labels.copy()
    for i in np.flatnonzero(labels >= 0):
        near = labels[max(0, i - SMOOTH) : i + SMOOTH + 1]
        out[i] = np.bincount(near[near >= 0]).argmax()
    return out


def to_turns(labels: np.ndarray, margins: np.ndarray, times: np.ndarray, gap_label: int) -> list:
    """Collapse per-window labels into turns: long pauses get gap_label, short pauses
    and very short turns are absorbed by a neighbouring speaker."""
    filled = labels.copy()
    i = 0
    while i < len(filled):
        if filled[i] < 0:
            j = i
            while j < len(filled) and filled[j] < 0:
                j += 1
            if (j - i - 1) * HOP + WIN >= LONG_GAP:
                filled[i:j] = gap_label
            i = j
        else:
            i += 1
    for i in range(1, len(filled)):
        if filled[i] < 0:
            filled[i] = filled[i - 1]
    first_real = filled[filled >= 0][0] if (filled >= 0).any() else 0
    filled[filled < 0] = first_real

    turns = []
    for i, lab in enumerate(filled):
        if turns and turns[-1]["label"] == lab:
            turns[-1]["last"] = i
        else:
            turns.append({"label": int(lab), "first": i, "last": i})

    def length(t):
        return (t["last"] - t["first"]) * HOP + WIN

    # Merge turns shorter than MIN_TURN into the longer neighbour, shortest first.
    while len(turns) > 1:
        idx = min(range(len(turns)), key=lambda k: length(turns[k]))
        if length(turns[idx]) >= MIN_TURN:
            break
        neighbours = [
            t for t in (turns[idx - 1] if idx else None, turns[idx + 1] if idx + 1 < len(turns) else None) if t
        ]
        speakers = [t for t in neighbours if t["label"] != gap_label]
        target = max(speakers or neighbours, key=length)
        target["first"] = min(target["first"], turns[idx]["first"])
        target["last"] = max(target["last"], turns[idx]["last"])
        del turns[idx]
        merged = [turns[0]]
        for t in turns[1:]:
            if t["label"] == merged[-1]["label"]:
                merged[-1]["last"] = t["last"]
            else:
                merged.append(t)
        turns = merged

    result = []
    for t in turns:
        start = result[-1]["end"] if result else float(times[t["first"]])
        end = float(times[t["last"]]) + WIN
        conf = float(np.mean(margins[t["first"] : t["last"] + 1]))
        result.append({"label": t["label"], "start": round(start, 2), "end": round(end, 2), "confidence": conf})
    return result


def find_voices(
    audio: np.ndarray, model_dir: str, threads: int, on_progress: Callable[[float], None] = lambda done: None
) -> List[dict]:
    """
    The recording's timeline: turns of {"start", "end", "group", "confidence"}, where "group" is the
    voice ("1" = heard the longest) or None for a long stretch without sound.
    """
    starts = window_starts(len(audio))
    speech = speech_mask(audio, starts)
    emb = fingerprints(audio, starts, speech, model_dir, threads, lambda done: on_progress(0.9 * done))
    if not len(emb):
        raise NoSpeech("No speech was found in the recording.")
    centroids = group_voices(emb)
    sims = emb @ centroids.T
    raw = np.full(len(starts), -1)
    raw[speech] = np.argmax(sims, axis=1)
    margins = np.zeros(len(starts))
    if len(centroids) > 1:
        top2 = np.sort(sims, axis=1)[:, -2:]
        margins[speech] = top2[:, 1] - top2[:, 0]
    gap = len(centroids)
    turns = to_turns(smooth(raw), margins, starts / SR, gap_label=gap)
    on_progress(1.0)
    return [
        {
            "start": t["start"],
            "end": t["end"],
            "group": None if t["label"] == gap else str(t["label"] + 1),
            "confidence": round(t["confidence"], 4),
        }
        for t in turns
    ]


def speech_turns(audio: np.ndarray, name: str = "teacher") -> List[dict]:
    """One speaker: the whole recording is ``name``, apart from long stretches without sound."""
    starts = window_starts(len(audio))
    speech = speech_mask(audio, starts)
    labels = np.where(speech, 0, -1)
    turns = to_turns(labels, np.zeros(len(starts)), starts / SR, gap_label=1)
    return [{"start": t["start"], "end": t["end"], "name": name if t["label"] == 0 else NO_SPEECH} for t in turns]


def name_turns(timeline: List[dict], labels: dict) -> List[dict]:
    """The author's tags on the timeline: each turn gets "name" teacher / translator / ... / "no speech"."""
    return [{**t, "name": NO_SPEECH if t["group"] is None else labels.get(t["group"], "other")} for t in timeline]


def voice_groups(timeline: List[dict]) -> List[dict]:
    """Each voice with its total minutes and its longest turns (for sample clips), biggest first."""
    groups = {}
    for t in timeline:
        if t["group"] is not None:
            groups.setdefault(t["group"], []).append(t)
    return [
        {
            "group": group,
            "minutes": round(sum(t["end"] - t["start"] for t in turns) / 60, 1),
            "longest": sorted(turns, key=lambda t: t["start"] - t["end"]),
        }
        for group, turns in sorted(groups.items(), key=lambda item: int(item[0]))
    ]


def sample_spans(turns: List[dict], count: int, length: float) -> List[tuple]:
    """
    (start, length) of up to ``count`` clips, in time order: one from the middle of each of the longest
    turns, and when a voice has fewer turns than that, more clips spread over its longest turns.
    """
    picks = {}  # turn index -> how many clips it gives
    for _ in range(count):
        fits = [i for i, t in enumerate(turns) if (picks.get(i, 0) + 1) * length <= t["end"] - t["start"]]
        unused = [i for i in fits if i not in picks]
        if not fits and not picks and turns:
            picks[0] = 1  # even the longest turn is shorter than one clip: take it whole
            break
        if not fits:
            break
        i = unused[0] if unused else fits[0]
        picks[i] = picks.get(i, 0) + 1
    spans = []
    for i, n in picks.items():
        t = turns[i]
        for k in range(n):
            middle = t["start"] + (t["end"] - t["start"]) * (k + 1) / (n + 1)
            start = max(t["start"], middle - length / 2)
            spans.append((round(start, 2), round(min(length, t["end"] - start), 2)))
    return sorted(spans)


def waveform(audio: np.ndarray, count: int = WAVE_LEVELS) -> dict:
    """
    How loud the recording is over time, for drawing its waveform: ``count`` values from 0 to 1, each the
    loudness (RMS) of an equal slice. Scaled so the loud parts of speech reach about 1, and gently
    curved (power 0.7) so quiet speech still shows while silence stays near 0.
    """
    count = max(1, min(count, len(audio) // 160))  # at least 10 ms per value
    slices = np.array_split(audio[: len(audio) // count * count], count)
    rms = np.sqrt(np.array([np.mean(part.astype(np.float64) ** 2) for part in slices]))
    top = np.percentile(rms, 98) or 1.0
    levels = np.clip(rms / top, 0, 1) ** 0.7
    return {"seconds_per_level": round(len(audio) / count / SR, 4), "levels": [round(float(v), 3) for v in levels]}
