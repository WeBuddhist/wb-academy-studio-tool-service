"""
Phase 2, step 1: suggest a course from the checked transcript, with the AI model (an OpenRouter model, by
default Google's Gemini: studio_tool_service/llm.py).

    1. Outline: one call with every line the teacher said (numbered as in transcript.json, with times)
       -> sections of lessons, each lesson a range of line numbers, with Tibetan titles. The ranges are
       checked, and small gaps between lessons closed; if they don't fit, the model is asked once more with
       what was wrong.
    2. Lessons: one call per lesson (a few at a time) -> summary, key points, practice questions, reflection
       questions. A reply with English (Latin-script) words in it is asked for once more.
    3. The draft, in the shape Studio keeps (sherab-custom-plugin course_from_recording CourseDraftSerializer).

The author's corrections (transcript_edits) are laid over the transcript first. Every reply is saved in
4-course/working/ as it arrives, so a retried job only asks for what is still missing.
"""

import bisect
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

SUGGESTION = "4-course/suggestion.json"
WORKING = "4-course/working"
OUTLINE_SHARE = 0.3  # of the step's progress; the lessons share the rest

PROMPTS = Path(__file__).parent / "prompts"
NO_SPEECH = "no speech"
TRANSLATOR = "translator"

# Studio's limits for the draft. Longer text is cut rather than refused, and extra items dropped; an outline
# with too many sections or lessons is asked for again.
TITLE_CHARS, SUMMARY_CHARS, ITEM_CHARS, OPTION_CHARS, EXPLANATION_CHARS = 255, 5000, 1000, 500, 2000
MAX_ITEMS = 10
MIN_OPTIONS, MAX_OPTIONS = 2, 6
MAX_SECTIONS, MAX_SECTION_LESSONS = 30, 50

# Lessons asked for at the same time: faster for a long teaching, without hammering OpenRouter.
LESSON_CALLS = 3

# An English (or other Latin-script) word: two letters or more. The course is written in Tibetan.
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


class NoTeacherText(ValueError):
    """The transcript has nothing the teacher said to make a course from."""


class OutlineProblem(ValueError):
    """The model's outline doesn't fit the transcript (the problems are in ``args[0]``)."""


def _object(title: str, properties: dict) -> dict:
    """A strict JSON Schema object: every property required, nothing else allowed."""
    return {
        "title": title,
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _list(items: dict) -> dict:
    return {"type": "array", "items": items}


TEXT, NUMBER = {"type": "string"}, {"type": "integer"}

OUTLINE_SCHEMA = _object("course_outline", {
    "course_title": TEXT,
    "sections": _list(_object("section", {
        "title": TEXT,
        "lessons": _list(_object("lesson", {"title": TEXT, "first_line": NUMBER, "last_line": NUMBER})),
    })),
})

LESSON_SCHEMA = _object("lesson_content", {
    "summary": TEXT,
    "key_points": _list(TEXT),
    "mcq": _list(_object("question", {
        "question": TEXT, "options": _list(TEXT), "answer": NUMBER, "explanation": TEXT,
    })),
    "reflection": _list(TEXT),
})


def prompt(name: str, **values) -> str:
    return Template((PROMPTS / f"{name}.md").read_text(encoding="utf-8")).substitute(**values)


def clock(seconds: float) -> str:
    total = int(round(seconds))
    hours, minutes, secs = total // 3600, total % 3600 // 60, total % 60
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def teacher_lines(lines: List[dict], edits: Dict[str, str]) -> List[dict]:
    """What the teacher said, with the author's corrections: [{index, start, end, text}] (index in transcript.json)."""
    said = []
    for index, line in enumerate(lines):
        if line.get("speaker") != "teacher":
            continue
        text = (edits.get(str(index)) or line.get("text") or "").strip()
        if text:
            said.append({"index": index, "start": line["start"], "end": line["end"], "text": text})
    return said


def recording_details(details: dict, lines: List[dict]) -> str:
    """The author's details about the recording, for both prompts."""
    rows = [f"Teacher: {details.get('teacher') or 'unknown'}"]
    if details.get("text"):
        rows.append(f"Text being taught: {details['text']}")
    if details.get("place") or details.get("year"):
        rows.append(f"Place and year: {', '.join(str(v) for v in (details.get('place'), details.get('year')) if v)}")
    if details.get("multiple_speakers"):
        translator = details.get("translator_language") or "another language"
        rows.append(f"The teacher speaks Tibetan and a translator interprets into {translator}; only the teacher is "
                    "transcribed.")
    if lines:
        rows.append(f"Length of the recording: {round(max(line['end'] for line in lines) / 60)} minutes")
    return "\n".join(rows)


def passage_starts(lines: List[dict], said: List[dict]) -> List[int]:
    """
    The lines a lesson may start on: the first line of each passage of the teacher's. With a translator, the
    teacher says a passage, then the translator translates it; a passage starts where the teacher speaks
    again after the translator (or chanting, or anyone else), and silences don't end one. Starting lessons
    only there keeps every passage in the same lesson as its translation. With no translator, every line the
    teacher said is a start.
    """
    if not any(line.get("speaker") == TRANSLATOR for line in lines):
        return [line["index"] for line in said]
    said_indices = {line["index"] for line in said}
    starts, new_passage = [], True
    for index, line in enumerate(lines):
        speaker = line.get("speaker")
        if speaker == NO_SPEECH:
            continue
        if speaker != "teacher":
            new_passage = True
        elif new_passage and index in said_indices:
            starts.append(index)
            new_passage = False
    return starts


def _nearest(starts: List[int], index: int) -> int:
    """The passage start closest to line ``index`` (the earlier one when two are as close)."""
    at = bisect.bisect_left(starts, index)
    before = starts[at - 1] if at > 0 else None
    after = starts[at] if at < len(starts) else None
    if after is None or (before is not None and index - before <= after - index):
        return before
    return after


def _start_on_passages(sections: List[dict], starts: List[int]) -> List[dict]:
    """
    Move each lesson's first line to the start of a passage (see passage_starts). A lesson that lies wholly
    inside one passage of the lesson before can't start anywhere of its own: it joins that lesson.
    """
    previous = None
    for section in sections:
        kept = []
        for lesson in section["lessons"]:
            start = _nearest(starts, lesson["first_line"])
            if previous is not None and start <= previous:
                later = starts[bisect.bisect_right(starts, previous)] if starts[-1] > previous else None
                if later is None or later > lesson["last_line"]:
                    log.info("Lesson %r lies inside one passage of the lesson before; joined to it", lesson["title"])
                    continue
                start = later
            lesson["first_line"] = previous = start
            kept.append(lesson)
        section["lessons"] = kept
    return [section for section in sections if section["lessons"]]


def check_outline(outline: dict, said: List[dict], starts: Optional[List[int]] = None) -> dict:
    """
    The outline, made to fit the transcript: each lesson starts on a line the teacher said, at the start of
    one of the teacher's passages (``starts``, see passage_starts; default: any line the teacher said), and
    runs up to the next lesson's first line (so every line between, the translator's too, belongs to one
    lesson); the first lesson starts at the teacher's first line and the last ends at the teacher's last
    line. Raises OutlineProblem if the ranges overlap, go backwards or fall outside.
    """
    indices = [line["index"] for line in said]
    problems, sections = [], []
    previous_last = -1
    if len(outline.get("sections") or []) > MAX_SECTIONS:
        problems.append(f"there are {len(outline['sections'])} sections; a course can have at most {MAX_SECTIONS}")
    for s_number, section in enumerate(outline.get("sections") or [], start=1):
        lessons = []
        if len(section.get("lessons") or []) > MAX_SECTION_LESSONS:
            problems.append(f"section {s_number} has {len(section['lessons'])} lessons; a section can have at most "
                            f"{MAX_SECTION_LESSONS}")
        for l_number, lesson in enumerate(section.get("lessons") or [], start=1):
            name = f"section {s_number}, lesson {l_number}"
            try:
                first, last = int(lesson["first_line"]), int(lesson["last_line"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{name} has no line numbers")
                continue
            # Onto lines the teacher said: the first one at or after `first`, the last one at or before `last`.
            at = bisect.bisect_left(indices, first)
            to = bisect.bisect_right(indices, last) - 1
            if first > last or at >= len(indices) or to < 0 or at > to:
                problems.append(f"{name} ({first}-{last}) has none of the teacher's lines")
                continue
            first, last = indices[at], indices[to]
            if first <= previous_last:
                problems.append(f"{name} ({first}-{last}) overlaps the lesson before it or goes back")
                continue
            previous_last = last
            lessons.append({"title": _cut(lesson.get("title"), TITLE_CHARS) or f"{l_number}", "first_line": first,
                            "last_line": last})
        if lessons:
            sections.append({"title": _cut(section.get("title"), TITLE_CHARS) or f"{s_number}", "lessons": lessons})
    if not sections and not problems:
        problems.append("there are no lessons")
    if problems:
        raise OutlineProblem(problems)
    if starts:
        sections = _start_on_passages(sections, starts)

    # Close the gaps: lines between two lessons go to the one before; the ends stretch to the first/last line.
    flat = [lesson for section in sections for lesson in section["lessons"]]
    flat[0]["first_line"] = indices[0]
    for before, after in zip(flat, flat[1:]):
        before["last_line"] = after["first_line"] - 1
    flat[-1]["last_line"] = indices[-1]
    return {"course_title": _cut(outline.get("course_title"), TITLE_CHARS), "sections": sections}


def listing(lines: List[dict], said: List[dict]) -> str:
    """
    The teacher's lines for the outline prompt, numbered and timed, with a "(translator)" mark wherever the
    translator speaks, so the model sees where each passage of the teaching is translated.
    """
    by_index = {line["index"]: line for line in said}
    rows, translator = [], False
    for index, line in enumerate(lines):
        if line.get("speaker") == TRANSLATOR:
            translator = True
        elif index in by_index:
            if translator and rows:
                rows.append("(translator)")
            translator = False
            rows.append(f"{index} [{clock(line['start'])}] {by_index[index]['text']}")
    return "\n".join(rows)


def lesson_times(outline: dict, lines: List[dict]) -> None:
    """
    Add each lesson's start and end (seconds) and length: it starts with its first line and ends where the
    next lesson starts, so the translator's reply to its last line stays in it. The first lesson starts with
    the first speech of the recording (so opening prayers and chanting before the teacher's first words are
    in it), and the last ends with the last.
    """
    flat = [lesson for section in outline["sections"] for lesson in section["lessons"]]
    speech = [line for line in lines if line.get("speaker") != NO_SPEECH]
    speech_start = min((float(line["start"]) for line in speech), default=0.0)
    speech_end = max((line["end"] for line in speech), default=0.0)
    for number, lesson in enumerate(flat):
        start = float(lines[lesson["first_line"]]["start"])
        lesson["start"] = round(min(start, speech_start) if number == 0 else start, 2)
        following = flat[number + 1] if number + 1 < len(flat) else None
        if following:
            end = lines[following["first_line"]]["start"]
        else:
            end = max(speech_end, lines[lesson["last_line"]]["end"])
        lesson["end"] = round(max(float(end), lesson["start"] + 0.1), 2)
        lesson["minutes"] = round((lesson["end"] - lesson["start"]) / 60, 1)


def _cut(text, limit: int) -> str:
    return str(text or "").strip()[:limit]


def _texts(items, limit: int) -> List[str]:
    texts = [_cut(item, limit) for item in (items or []) if isinstance(item, str)]
    return [text for text in texts if text][:MAX_ITEMS]


def clean_lesson(reply: dict) -> dict:
    """The model's lesson, within Studio's limits; a question with repeated options or no right one is left out."""
    questions = []
    for item in reply.get("mcq") or []:
        options = [_cut(option, OPTION_CHARS) for option in item.get("options") or [] if isinstance(option, str)]
        try:
            answer = int(item.get("answer"))
        except (TypeError, ValueError):
            continue
        question = _cut(item.get("question"), ITEM_CHARS)
        if (not question or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS or "" in options
                or len(set(options)) != len(options) or not 0 <= answer < len(options)):
            log.info("Left out a question the model wrote that doesn't work: %s", item)
            continue
        questions.append({"question": question, "options": options, "answer": answer,
                          "explanation": _cut(item.get("explanation"), EXPLANATION_CHARS)})
    return {
        "summary": _cut(reply.get("summary"), SUMMARY_CHARS),
        "key_points": _texts(reply.get("key_points"), ITEM_CHARS),
        "mcq": questions[:MAX_ITEMS],
        "reflection": _texts(reply.get("reflection"), ITEM_CHARS),
    }


def latin_words(content: dict) -> List[str]:
    """The English (Latin-script) words in a lesson's texts, which should all be Tibetan."""
    texts = [content["summary"], *content["key_points"], *content["reflection"]]
    for question in content["mcq"]:
        texts += [question["question"], question["explanation"], *question["options"]]
    return sorted({word for text in texts for word in LATIN_WORD.findall(text)})


def build_draft(outline: dict, contents: Dict[str, dict]) -> dict:
    """The draft Studio keeps: ids s1, s1-l1, ...; every lesson kept to start with."""
    sections = []
    for s_number, section in enumerate(outline["sections"], start=1):
        lessons = []
        for l_number, lesson in enumerate(section["lessons"], start=1):
            content = contents[f"{lesson['first_line']}-{lesson['last_line']}"]
            lessons.append({
                "id": f"s{s_number}-l{l_number}",
                "title": lesson["title"],
                "include": True,
                "first_line": lesson["first_line"],
                "last_line": lesson["last_line"],
                "start": lesson["start"],
                "end": lesson["end"],
                "minutes": lesson["minutes"],
                **content,
            })
        sections.append({"id": f"s{s_number}", "title": section["title"], "lessons": lessons})
    return {"version": 1, "course_title": outline["course_title"], "sections": sections}


class Suggester:
    """Runs the step for one job. ``key(path)`` gives a path's S3 key in the job's folder."""

    def __init__(self, llm, storage, key: Callable[[str], str], details: dict):
        self.llm, self.storage, self.key, self.details = llm, storage, key, details
        self.system = (PROMPTS / "system.md").read_text(encoding="utf-8")

    def run(self, lines: List[dict], edits: Dict[str, str], progress: Callable[[float], None]) -> dict:
        said = teacher_lines(lines, edits)
        if not said:
            raise NoTeacherText()
        details = recording_details(self.details, lines)

        outline = self.outline(lines, said, details)
        lesson_times(outline, lines)
        progress(OUTLINE_SHARE)

        lessons = [(section, lesson) for section in outline["sections"] for lesson in section["lessons"]]
        contents = {}
        # A few lessons at a time; each reply is saved as it arrives (see lesson()), so a retry after a failure
        # only asks for the rest.
        with ThreadPoolExecutor(max_workers=LESSON_CALLS) as pool:
            asked = {
                pool.submit(self.lesson, outline["course_title"], section, lesson, said, details): lesson
                for section, lesson in lessons
            }
            try:
                for number, answer in enumerate(as_completed(asked), start=1):
                    lesson = asked[answer]
                    contents[f"{lesson['first_line']}-{lesson['last_line']}"] = answer.result()
                    # Rounded, so the last lesson makes exactly 1.0 (100 %), not 0.9999...
                    progress(round(OUTLINE_SHARE + (1 - OUTLINE_SHARE) * number / len(lessons), 6))
            except BaseException:
                # A lesson failed, or Studio stopped the job: ask for no more (the ones under way finish).
                pool.shutdown(cancel_futures=True)
                raise

        draft = build_draft(outline, contents)
        self.storage.write_json(self.key(SUGGESTION), draft)
        log.info("Course suggested: %s sections, %s lessons; model tokens in %s, out %s", len(draft["sections"]),
                 len(lessons), self.llm.tokens_in, self.llm.tokens_out)
        return draft

    def outline(self, lines: List[dict], said: List[dict], details: str) -> dict:
        starts = passage_starts(lines, said)
        saved_key = self.key(f"{WORKING}/outline.json")
        saved = self.storage.read_json(saved_key)
        if saved:
            try:
                return check_outline(saved, said, starts)
            except OutlineProblem:
                log.warning("The saved outline no longer fits the transcript; asking the model again")

        request = prompt("outline", details=details, lines=listing(lines, said))
        reply = self.llm.json(request, OUTLINE_SCHEMA, self.system)
        try:
            outline = check_outline(reply, said, starts)
        except OutlineProblem as problem:
            log.warning("The model's outline didn't fit the transcript (%s); asking once more", problem.args[0])
            problems = "\n".join(f"- {text}" for text in problem.args[0])
            again = (f"{request}\n\n## Your previous outline didn't fit\n\n{problems}\n\n"
                     "Write the outline again, keeping to the rules.")
            outline = check_outline(self.llm.json(again, OUTLINE_SCHEMA, self.system), said, starts)
        self.storage.write_json(saved_key, outline)
        return outline

    def lesson(self, course_title: str, section: dict, lesson: dict, said: List[dict], details: str) -> dict:
        saved_key = self.key(f"{WORKING}/lesson-{lesson['first_line']}-{lesson['last_line']}.json")
        saved = self.storage.read_json(saved_key)
        if saved:
            return clean_lesson(saved)
        text = "\n".join(line["text"] for line in said if lesson["first_line"] <= line["index"] <= lesson["last_line"])
        request = prompt("lesson", details=details, course_title=course_title or "-", section_title=section["title"],
                         lesson_title=lesson["title"], text=text)
        content = clean_lesson(self.llm.json(request, LESSON_SCHEMA, self.system))
        english = latin_words(content)
        if english:
            log.warning("Lesson %r had English words (%s); asking once more", lesson["title"], ", ".join(english))
            again = (f"{request}\n\n## Your previous answer had English words\n\n{', '.join(english)}\n\n"
                     "Write it again with everything in Tibetan: no English or other words in Latin letters.")
            content = clean_lesson(self.llm.json(again, LESSON_SCHEMA, self.system))
        self.storage.write_json(saved_key, content)
        return content
