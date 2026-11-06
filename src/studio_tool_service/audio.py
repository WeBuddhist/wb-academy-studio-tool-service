"""
Reading and writing audio. Uploads are decoded once with ffmpeg into a mono 16-bit WAV working copy
(both channels mixed, since some recorders leave one silent); every step then reads only the part it
needs from that file, so a long recording never has to fit in memory twice.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def duration(path: Path) -> float:
    """Length of any audio (or video) file in seconds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def to_wav(src: Path, dest: Path, sample_rate: int) -> Path:
    """Decode any audio (or video) file to a mono 16-bit WAV at ``sample_rate``."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        check=True,
    )
    return dest


def read(path: Path, start: float = 0.0, end: float = None) -> np.ndarray:
    """Seconds ``start`` to ``end`` (default: the end of the file) as float32 samples."""
    with sf.SoundFile(str(path)) as f:
        f.seek(min(int(start * f.samplerate), f.frames))
        frames = -1 if end is None else max(0, int((end - start) * f.samplerate))
        return f.read(frames, dtype="float32")


def cut_mp3(src: Path, dest: Path, start: float, length: float) -> Path:
    """A short mono MP3 clip of ``src``, e.g. a sample of one voice."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{length:.2f}",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(dest),
        ],
        check=True,
    )
    return dest
