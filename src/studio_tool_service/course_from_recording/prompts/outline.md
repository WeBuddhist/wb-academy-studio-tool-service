## The recording

$details

## The task

Below is everything the teacher said, one numbered line per short piece of speech, with the time it starts
in the recording. (Lines of the translator, chanting and silences are left out, so the numbers jump.) Where
the translator speaks, there is a `(translator)` line: the teacher's passage just before it is translated
there.

Divide the teaching into a course outline:

- **Sections**: the main themes of the teaching, in the order they are taught.
- **Lessons** inside each section: one topic each, following the natural breaks in the teaching, where the
  teacher moves on to something new. A lesson is a range of lines, `first_line` to `last_line` (the line
  numbers below).

Rules:

- Decide how many sections and lessons the teaching needs; there is no fixed number. A short teaching may be
  one section; a lesson is usually 5 to 20 minutes of teaching (the times show this). Don't make a lesson of
  less than about 3 minutes unless the whole teaching is that short. The course can have at most 30
  sections, and a section at most 50 lessons.
- When there are `(translator)` lines, start each lesson on the first line after a `(translator)` line, so
  every passage stays in the same lesson as its translation.
- Every line belongs to exactly one lesson. Lessons follow each other in order: each lesson's `first_line`
  is after the previous lesson's `last_line`, and together they cover every line from the first to the last.
- Opening prayers, greetings or practical announcements can be a short lesson of their own: the course's
  author can switch it off.
- Titles are in Tibetan, short (a few words), and say what is taught, like the chapter titles of a Dharma
  book. `course_title` is the whole teaching's title, in Tibetan.

## The teacher's words

$lines
