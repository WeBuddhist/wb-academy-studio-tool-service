"""Suggesting a course from the transcript (course_from_recording/suggest.py), with the AI model replaced by a fake."""

import pytest

from studio_tool_service.course_from_recording import suggest
from studio_tool_service.course_from_recording.suggest import OutlineProblem, Suggester

# transcript.json: the teacher's lines (some to be corrected), the translator, silences.
LINES = [
    {"start": 0.0, "end": 20.0, "speaker": "teacher", "text": "ཀ།"},
    {"start": 20.0, "end": 40.0, "speaker": "teacher", "text": "ཁ།"},
    {"start": 40.0, "end": 60.0, "speaker": "teacher", "text": ""},  # Monlam heard nothing
    {"start": 60.0, "end": 100.0, "speaker": "translator", "text": None},
    {"start": 100.0, "end": 130.0, "speaker": "teacher", "text": "ག།"},
    {"start": 130.0, "end": 160.0, "speaker": "teacher", "text": "ང།"},
    {"start": 160.0, "end": 170.0, "speaker": "no speech", "text": None},
    {"start": 170.0, "end": 200.0, "speaker": "teacher", "text": "ཅ།"},
    {"start": 200.0, "end": 230.0, "speaker": "teacher", "text": "ཆ།"},
    {"start": 230.0, "end": 260.0, "speaker": "teacher", "text": "ཇ།"},
    {"start": 260.0, "end": 300.0, "speaker": "translator", "text": None},
    {"start": 300.0, "end": 400.0, "speaker": "no speech", "text": None},
]
DETAILS = {"teacher": "Geshe Tenzin", "text": "སྤྱོད་འཇུག", "multiple_speakers": True, "translator_language": "Ladakhi"}


def outline(*ranges):
    """An outline with one section per list of (first, last) ranges."""
    return {
        "course_title": "སྤྱོད་འཇུག",
        "sections": [
            {"title": f"ལེའུ་{number}",
             "lessons": [{"title": f"ཚན་{a}", "first_line": a, "last_line": b} for a, b in section]}
            for number, section in enumerate(ranges, start=1)
        ],
    }


LESSON = {
    "summary": "བསྡུས་དོན།",
    "key_points": ["གནད་ཀ", "གནད་ཁ", "གནད་ག"],
    "mcq": [{"question": "དྲི་བ།", "options": ["ཀ", "ཁ", "ག"], "answer": 1, "explanation": "རྒྱུ་མཚན།"}],
    "reflection": ["བསམ་ཀ", "བསམ་ཁ", "བསམ་ག"],
}


class FakeLLM:
    def __init__(self, *outlines):
        self.outlines, self.prompts = list(outlines), []
        self.tokens_in = self.tokens_out = 0

    def json(self, prompt, schema, system=""):
        self.prompts.append(prompt)
        assert "Tibetan" in system
        if schema is suggest.OUTLINE_SCHEMA:
            return self.outlines.pop(0)
        return dict(LESSON)


class MemoryStorage:
    def __init__(self):
        self.files = {}

    def read_json(self, key):
        return self.files.get(key)

    def write_json(self, key, data):
        self.files[key] = data


def key(path):
    return f"folder/{path}"


def run(model, storage=None, edits=None):
    storage = storage or MemoryStorage()
    done = []
    draft = Suggester(model, storage, key, DETAILS).run(LINES, edits or {}, done.append)
    return draft, storage, done


def test_the_teachers_lines_with_the_authors_corrections():
    said = suggest.teacher_lines(LINES, {"1": "ཁ་ཁ།", "2": "ཀ་ཁ་ག"})
    assert [(line["index"], line["text"]) for line in said] == [
        (0, "ཀ།"), (1, "ཁ་ཁ།"), (2, "ཀ་ཁ་ག"), (4, "ག།"), (5, "ང།"), (7, "ཅ།"), (8, "ཆ།"), (9, "ཇ།")
    ]


def test_the_outline_is_fitted_to_the_teachers_lines_and_its_gaps_closed():
    said = suggest.teacher_lines(LINES, {})
    fitted = suggest.check_outline(outline([(0, 0)], [(3, 6), (8, 9)]), said)
    ranges = [(lesson["first_line"], lesson["last_line"]) for s in fitted["sections"] for lesson in s["lessons"]]
    # Lesson 2 starts on the teacher's next line (4, not the translator at 3); every line up to the next
    # lesson belongs to the one before (1-3 to lesson 1, 6-7 to lesson 2).
    assert ranges == [(0, 3), (4, 7), (8, 9)]


@pytest.mark.parametrize("ranges, problem", [
    ([[(0, 4), (4, 9)]], "overlaps"),
    ([[(5, 9), (0, 4)]], "goes back"),
    ([[(0, 4), (20, 30)]], "none of the teacher's lines"),
    ([[(3, 3)]], "none of the teacher's lines"),
    ([[]], "no lessons"),
])
def test_an_outline_that_doesnt_fit_is_a_problem(ranges, problem):
    with pytest.raises(OutlineProblem) as error:
        suggest.check_outline(outline(*ranges), suggest.teacher_lines(LINES, {}))
    assert any(problem in text for text in error.value.args[0])


def test_each_lesson_runs_to_the_next_so_the_translator_stays_with_it():
    fitted = suggest.check_outline(outline([(0, 2), (4, 9)]), suggest.teacher_lines(LINES, {}))
    suggest.lesson_times(fitted, LINES)
    first, second = fitted["sections"][0]["lessons"]
    assert (first["start"], first["end"], first["minutes"]) == (0.0, 100.0, 1.7)
    # The last lesson ends with the last speech (the translator at 300 s), not the silence after it.
    assert (second["start"], second["end"]) == (100.0, 300.0)


def test_questions_that_dont_work_are_left_out_and_text_kept_to_studios_limits():
    reply = {
        "summary": "ཀ" * 6000,
        "key_points": ["ཀ", "", 5] + ["ཁ"] * 20,
        "mcq": [
            {"question": "ok", "options": ["ཀ", "ཁ"], "answer": 0, "explanation": ""},
            {"question": "same options", "options": ["ཀ", "ཀ ", "ཁ"], "answer": 0, "explanation": ""},
            {"question": "answer outside", "options": ["ཀ", "ཁ"], "answer": 2, "explanation": ""},
            {"question": "one option", "options": ["ཀ"], "answer": 0, "explanation": ""},
            {"question": "", "options": ["ཀ", "ཁ"], "answer": 0, "explanation": ""},
        ],
        "reflection": ["བསམ།"],
    }
    lesson = suggest.clean_lesson(reply)
    assert len(lesson["summary"]) == suggest.SUMMARY_CHARS
    assert len(lesson["key_points"]) == 10 and lesson["key_points"][0] == "ཀ"
    assert [question["question"] for question in lesson["mcq"]] == ["ok"]
    assert lesson["reflection"] == ["བསམ།"]


def test_suggests_a_draft_in_the_shape_studio_keeps():
    model = FakeLLM(outline([(0, 2)], [(4, 9)]))

    draft, storage, done = run(model, edits={"1": "ཁ་ཁ།"})

    assert draft["version"] == 1 and draft["course_title"] == "སྤྱོད་འཇུག"
    assert [section["id"] for section in draft["sections"]] == ["s1", "s2"]
    lessons = [lesson for section in draft["sections"] for lesson in section["lessons"]]
    assert [lesson["id"] for lesson in lessons] == ["s1-l1", "s2-l1"]
    assert [(lesson["first_line"], lesson["last_line"]) for lesson in lessons] == [(0, 3), (4, 9)]
    assert all(lesson["include"] for lesson in lessons)
    assert lessons[0]["mcq"] == LESSON["mcq"] and lessons[0]["summary"] == "བསྡུས་དོན།"
    assert storage.files[key(suggest.SUGGESTION)] == draft
    # One outline and one call per lesson; the outline sees the numbered, timed lines and the corrections.
    assert len(model.prompts) == 3
    assert "1 [0:20] ཁ་ཁ།" in model.prompts[0] and "Ladakhi" in model.prompts[0]
    # The lessons are asked for a few at a time, so in any order.
    assert any("ཁ་ཁ།" in asked and "ཚན་0" in asked for asked in model.prompts[1:])
    assert done[0] == suggest.OUTLINE_SHARE and done[-1] == 1.0


def test_a_retry_only_asks_for_what_is_missing():
    first = FakeLLM(outline([(0, 2), (4, 9)]))
    _, storage, _ = run(first)
    assert key("4-course/working/lesson-0-3.json") in storage.files
    del storage.files[key("4-course/working/lesson-4-9.json")]

    again = FakeLLM()
    run(again, storage)

    assert len(again.prompts) == 1  # only the lesson whose reply was lost


def test_an_outline_that_doesnt_fit_is_asked_for_once_more():
    model = FakeLLM(outline([(0, 4), (2, 9)]), outline([(0, 4), (5, 9)]))
    draft, _, _ = run(model)
    assert "overlaps" in model.prompts[1]
    # The second lesson (5-9) starts mid-passage: it moves back to the passage's start (4), after the translator.
    assert [lesson["first_line"] for lesson in draft["sections"][0]["lessons"]] == [0, 4]


def test_and_then_gives_up():
    model = FakeLLM(outline([(0, 4), (2, 9)]), outline([(9, 9), (0, 1)]))
    with pytest.raises(OutlineProblem):
        run(model)


def test_no_teacher_words_no_course():
    with pytest.raises(suggest.NoTeacherText):
        Suggester(FakeLLM(), MemoryStorage(), key, DETAILS).run(
            [{"start": 0.0, "end": 9.0, "speaker": "translator", "text": None}], {}, lambda done: None
        )


# ---------- each passage of the teacher's stays with its translation ----------

def test_a_passage_starts_where_the_teacher_speaks_again_after_the_translator():
    said = suggest.teacher_lines(LINES, {})
    # 0-2 is one passage (translated at 3); 4-9 is the next (a silence at 6 doesn't end it; translated at 10).
    assert suggest.passage_starts(LINES, said) == [0, 4]


def test_without_a_translator_any_line_can_start_a_lesson():
    lines = [line for line in LINES if line["speaker"] != "translator"]
    said = suggest.teacher_lines(lines, {})
    assert suggest.passage_starts(lines, said) == [line["index"] for line in said]


def test_the_model_sees_where_the_translator_speaks():
    listing = suggest.listing(LINES, suggest.teacher_lines(LINES, {})).splitlines()
    assert listing == [
        "0 [0:00] ཀ།", "1 [0:20] ཁ།", "(translator)", "4 [1:40] ག།", "5 [2:10] ང།", "7 [2:50] ཅ།", "8 [3:20] ཆ།",
        "9 [3:50] ཇ།",
    ]


def test_a_break_in_the_middle_of_a_passage_moves_to_its_start():
    said = suggest.teacher_lines(LINES, {})
    starts = suggest.passage_starts(LINES, said)
    fitted = suggest.check_outline(outline([(0, 5), (7, 9)]), said, starts)
    ranges = [(lesson["first_line"], lesson["last_line"]) for s in fitted["sections"] for lesson in s["lessons"]]
    # Breaking at 7 would split the passage 4-9 from its translation at 10: the break moves back to 4.
    assert ranges == [(0, 3), (4, 9)]


def test_a_lesson_inside_one_passage_joins_the_lesson_before():
    said = suggest.teacher_lines(LINES, {})
    starts = suggest.passage_starts(LINES, said)
    fitted = suggest.check_outline(outline([(0, 2)], [(4, 5), (7, 9)]), said, starts)
    lessons = [lesson for s in fitted["sections"] for lesson in s["lessons"]]
    assert [(lesson["first_line"], lesson["last_line"]) for lesson in lessons] == [(0, 3), (4, 9)]
    assert [lesson["title"] for lesson in lessons] == ["ཚན་0", "ཚན་4"]


def test_an_outline_with_more_than_studio_takes_is_a_problem():
    too_many = {"course_title": "", "sections": [
        {"title": "ལེའུ།", "lessons": [{"title": "ཚན།", "first_line": 0, "last_line": 0}]}
    ] * (suggest.MAX_SECTIONS + 1)}
    with pytest.raises(OutlineProblem) as problem:
        suggest.check_outline(too_many, suggest.teacher_lines(LINES, {}))
    assert any("at most 30" in text for text in problem.value.args[0])

    crowded = outline([(index, index) for index in range(suggest.MAX_SECTION_LESSONS + 1)])
    with pytest.raises(OutlineProblem) as problem:
        suggest.check_outline(crowded, suggest.teacher_lines(LINES, {}))
    assert any("at most 50" in text for text in problem.value.args[0])


class EnglishOnceLLM(FakeLLM):
    """Writes the first lesson with an English word in it, then in Tibetan when asked again."""

    def json(self, prompt, schema, system=""):
        reply = super().json(prompt, schema, system)
        if schema is suggest.LESSON_SCHEMA and "English words" not in prompt:
            reply["reflection"] = ["daily བསམ་ཞིབ།"]
        return reply


def test_a_lesson_with_english_words_is_asked_for_again():
    model = EnglishOnceLLM(outline([(0, 9)]))

    draft, _, _ = run(model)

    assert draft["sections"][0]["lessons"][0]["reflection"] == LESSON["reflection"]
    assert len(model.prompts) == 3 and "daily" in model.prompts[2]


def test_the_first_lesson_starts_with_the_opening_chanting():
    lines = [{"start": 0.0, "end": 12.0, "speaker": "chanting", "text": "ཨོཾ།"}] + [
        {**line, "start": line["start"] + 12.0, "end": line["end"] + 12.0} for line in LINES
    ]
    fitted = suggest.check_outline(outline([(1, 3), (5, 10)]), suggest.teacher_lines(lines, {}))
    suggest.lesson_times(fitted, lines)
    first, second = fitted["sections"][0]["lessons"]
    assert first["start"] == 0.0  # the chanting, not the teacher's first words at 12 s
    assert second["start"] == 112.0
