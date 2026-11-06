"""
Turning Monlam's replies into the transcript, and the rule-based Tibetan spelling fixes. Ported from the
pipeline's transcribe.py (build_transcript) and correct_transcript.py (rule_fix, the automatic part;
hand corrections come later, on the transcript screen).
"""

import re
from typing import Dict, List

from .pieces import warnings_for


def fmt(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def build_lines(plan: List[dict], results: Dict[str, dict], turns: List[dict]) -> List[dict]:
    """One line per Monlam segment, with times in the original recording, plus a line for every turn
    that was not transcribed (translator, chanting, ...), all in time order. (The pipeline script left
    out turns after the last transcribed piece, for its partial --limit-minutes runs; the service always
    transcribes the whole recording, so every turn is listed.)"""
    lines = []
    for piece in plan:
        result = results[piece["id"]].get("result") or {}
        warns = warnings_for(piece, result)
        segs = result.get("segments") or [
            {"start": 0, "end": piece["end"] - piece["start"], "text": result.get("text") or ""}
        ]
        for k, s in enumerate(segs):
            lines.append(
                {
                    "start": round(piece["start"] + s["start"], 2),
                    "end": round(min(piece["start"] + s["end"], piece["end"]), 2),
                    "speaker": piece["label"],
                    "text": (s["text"] or "").strip(),
                    "piece": piece["id"],
                    "warnings": warns if k == 0 else [],
                    "near_switch": piece["segment_edge"] and (k == 0 or k == len(segs) - 1),
                }
            )
    sent = {p["segment"] for p in plan}
    for idx, turn in enumerate(turns):
        if idx not in sent:
            lines.append(
                {
                    "start": turn["start"],
                    "end": turn["end"],
                    "speaker": turn["name"],
                    "text": None,
                    "piece": None,
                    "warnings": [],
                    "near_switch": False,
                }
            )
    lines.sort(key=lambda line: line["start"])
    return lines


def as_text(lines: List[dict], heading: str) -> str:
    out = [heading, "# ⚠ = please listen and check   ⓘ = at a speaker switch", ""]
    for line in lines:
        when = f"[{fmt(line['start'])} - {fmt(line['end'])}]"
        if line["text"] is None:
            out.append(f"{when}  ({line['speaker']} - not transcribed)")
            continue
        mark = "⚠" if line["warnings"] else ("ⓘ" if line["near_switch"] else " ")
        out.append(f"{when} {mark} {line['text']}")
        out.extend(f"{' ' * 30}⚠ {w}" for w in line["warnings"])
    return "\n".join(out) + "\n"


# ---------- Tibetan spelling (rule-based) ----------

BASE = set(chr(c) for c in range(0x0F40, 0x0F6D))  # Tibetan base consonants
VOWEL_SIGNS = set(chr(c) for c in range(0x0F71, 0x0F85))


def particle_for(prev: str) -> str:
    """Correct form of the gi/kyi/gyi particle after syllable `prev`."""
    marks = [k for k, ch in enumerate(prev) if ch in VOWEL_SIGNS or "ྐ" <= ch <= "ྼ"]
    if marks:  # letters after the vowel sign / subscript are the suffixes
        suffixes = [ch for ch in prev[marks[-1] + 1 :] if ch in BASE]
    else:  # no vowel sign: work out root + suffixes from the letter count
        letters = [ch for ch in prev if ch in BASE]
        if len(letters) <= 2:
            suffixes = letters[1:]
        elif letters[-1] == "ས" and letters[-2] in "གངབམ":
            suffixes = letters[-2:]
        else:
            suffixes = letters[-1:]
    final = suffixes[-1] if suffixes else ""
    if final in ("", "འ"):
        return "གི"  # open syllable
    if final in "དབས":
        return "ཀྱི"
    if final in "ནམརལ":
        return "གྱི"
    return "གི"  # ག, ང


# Spoken forms Monlam spells inconsistently -> one written form.
SPOKEN_FORMS = [
    (r"ཏ[ོི]ག་ཙི", "ཏོག་ཙམ"),
    (r"(?<=་)རཱ(?=[་།\s]|$)", "རེད"),  # ཡོད་རཱ → ཡོད་རེད
    (r"ཟེར་གོ་རེ(ད)?(?=[་།\s]|$)", "ཟེར་དགོས་རེད"),
]

PARTICLES = {p + end: end for p in ("ཀི", "གི", "ཀྱི", "གྱི") for end in ("", "ས", "ན")}
NOT_PARTICLES = {"སྤྱང་ཀི"}  # words that end in ཀི but are not the particle (སྤྱང་ཀི = wolf)


def rule_fix(text: str) -> str:
    """Only clear, mechanical errors: the gi/kyi/gyi particle (chosen by the previous syllable's final
    letter), "མྱི" -> "མི", spoken forms, and stray punctuation left by Monlam."""
    words = re.split(r"(་)", text)  # keep the tshegs so we can rebuild exactly
    for i, w in enumerate(words):
        core = re.match(r"\s*([^\s།]*)", w).group(1)  # the syllable, without a following ། or space
        if core in PARTICLES:
            j = i - 1
            while j >= 0 and words[j] in ("་", ""):
                j -= 1
            prev = re.split(r"[\s།]", words[j])[-1] if j >= 0 else ""
            if f"{prev}་{core}" in NOT_PARTICLES:
                continue
            words[i] = w.replace(core, particle_for(prev) + PARTICLES[core], 1)
        elif core == "མྱི":
            words[i] = w.replace(core, "མི", 1)
    out = "".join(words)
    for wrong, right in SPOKEN_FORMS:
        out = re.sub(wrong, right, out)
    out = re.sub(r"(?<=[^ང\s])་(?=།)", "", out)  # "རེད་།" -> "རེད།" (but "ང་།" keeps its tsheg)
    out = re.sub(r"།\s*་", "། ", out)  # "།་" -> "། "
    out = re.sub(r"(།\s*){2,}", "། ", out)  # runs of shads -> one
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+།", "།", out)
    return out


def corrected(lines: List[dict]) -> List[dict]:
    """The lines with Tibetan spelling fixed; the reply as Monlam gave it stays in "original"."""
    return [
        {**line, "original": line["text"], "text": rule_fix(line["text"]) if line["text"] else line["text"]}
        for line in lines
    ]
