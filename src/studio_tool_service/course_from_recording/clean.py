"""
Remove the background static, using the recording's own static as the noise sample. Ported from the
pipeline's denoise.py with its defaults (the "smooth" method, "medium" strength); the maths is
unchanged. The difference: the recording is read and written block by block, so a long recording never
has to fit in memory (the script held two full copies, ~3.8 GB for three hours).

    1. Take up to two minutes of pure static: the long "no speech" stretches of the timeline, or,
       when there are none, the quiet pauses between words.
    2. Cut out the low rumble (below LOW_CUT Hz) - it is below the voices.
    3. Learn the static's level at every pitch.
    4. Everywhere, turn each pitch down gradually by how far it is above the static.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, istft, sosfiltfilt, stft

from .. import audio

SR = 44100
LOW_CUT = 80  # Hz; the rumble sits at 40-80 Hz
NFFT = 2048
HOP = NFFT // 4
REDUCE_DB = 14  # "medium": how many dB the static is turned down
SMOOTH = 0.98  # how much each frame's estimate leans on the previous one
BLOCK = 60.0  # seconds cleaned at a time
PAD = 2.0  # extra seconds on each side of a block, so block edges don't click
NOISE_SECONDS = 120.0
MIN_QUIET = 3.0  # a "no speech" stretch must be this long to sample static from
EDGE = 0.5  # stay this far clear of a stretch's edges


class NoStatic(Exception):
    """No quiet stretch long enough to learn the static from."""


def quiet_spans(timeline: List[dict], pauses: np.ndarray) -> List[Tuple[float, float]]:
    """
    (start, end) of stretches that are only static, longest first: the "no speech" turns; in a recording
    without long silences (often a short one), the pauses of a second or more; failing that, every
    pause between words.
    """
    spans = [(t["start"] + EDGE, t["end"] - EDGE) for t in timeline if t["name"] == "no speech"]
    spans = [(a, b) for a, b in spans if b - a >= MIN_QUIET]
    if not spans:
        spans = [(float(a) + 0.1, float(b) - 0.1) for a, b in pauses if b - a >= 1.0]
    if not spans:
        spans = [(float(a), float(b)) for a, b in pauses]
    return sorted(spans, key=lambda span: span[0] - span[1])


def noise_sample(wav: Path, spans: List[Tuple[float, float]]) -> np.ndarray:
    parts, total = [], 0.0
    for start, end in spans:
        if total >= NOISE_SECONDS:
            break
        length = min(end - start, NOISE_SECONDS - total)
        parts.append(audio.read(wav, start, start + length))
        total += length
    if not parts:
        raise NoStatic("No quiet stretch was found to learn the background noise from.")
    return np.concatenate(parts)


def highpass(x: np.ndarray) -> np.ndarray:
    return sosfiltfilt(butter(4, LOW_CUT, "highpass", fs=SR, output="sos"), x).astype(np.float32)


# A floor for the static's power: a recorder's digital silence (exact zeros) has none at some pitches,
# and dividing by zero there would fill the cleaned audio with NaN.
MIN_NOISE_POWER = 1e-12


def noise_power(noise: np.ndarray) -> np.ndarray:
    """The static's average power at every pitch."""
    _, _, z = stft(highpass(noise), SR, nperseg=NFFT, noverlap=NFFT - HOP)
    return np.maximum((np.abs(z) ** 2).mean(axis=1), MIN_NOISE_POWER)


def clean_block(x: np.ndarray, power_of_noise: np.ndarray) -> np.ndarray:
    """Turn each pitch down gradually by how far it is above the static (a Wiener-style filter with
    "decision-directed" smoothing), instead of an on/off gate - fewer watery or muffled side effects."""
    _, _, z = stft(highpass(x), SR, nperseg=NFFT, noverlap=NFFT - HOP)
    power = np.abs(z) ** 2
    floor = 10 ** (-REDUCE_DB / 20)
    gain = np.empty(power.shape, np.float32)
    prev = np.zeros(power.shape[0])
    for t in range(power.shape[1]):
        post = power[:, t] / power_of_noise
        prior = SMOOTH * prev + (1 - SMOOTH) * np.maximum(post - 1, 0)
        g = np.maximum(prior / (1 + prior), floor)
        gain[:, t] = g
        prev = g**2 * post
    _, y = istft(z * gain, SR, nperseg=NFFT, noverlap=NFFT - HOP)
    return y[: len(x)].astype(np.float32)


def clean_file(
    wav: Path,
    dest: Path,
    power_of_noise: Optional[np.ndarray],
    on_progress: Callable[[float], None] = lambda done: None,
) -> Path:
    """Clean the 44.1 kHz working WAV into a FLAC file (lossless, about half the size of WAV). Without a
    noise sample (``None``) the sound is only copied, with the rumble below LOW_CUT cut."""
    with sf.SoundFile(str(wav)) as src, sf.SoundFile(
        str(dest), "w", samplerate=SR, channels=1, format="FLAC", subtype="PCM_16"
    ) as out:
        n, pad, total = int(BLOCK * SR), int(PAD * SR), src.frames
        for a in range(0, total, n):
            lo, hi = max(0, a - pad), min(total, a + n + pad)
            src.seek(lo)
            x = src.read(hi - lo, dtype="float32")
            y = highpass(x) if power_of_noise is None else clean_block(x, power_of_noise)
            out.write(np.clip(y[a - lo : a - lo + min(n, total - a)], -1, 1))
            on_progress(min(a + n, total) / total)
    return dest
