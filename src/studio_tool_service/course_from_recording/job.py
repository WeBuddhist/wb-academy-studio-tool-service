"""
Runs one recording through the "course from a recording" steps.

    Phase 1, up to the transcript:
        More than one speaker: find the voices -> pause for the author's tags -> clean -> transcribe the teacher
        One speaker:           clean -> transcribe
    Phase 2, the course (each run starts from the author's click in Studio):
        suggest a course (the AI model, through OpenRouter) -> pause while the author shapes it
        cut the lesson audio -> Studio writes the course

What a run does follows from what is already done: an approved draft -> cut the audio and write the course;
a transcript -> suggest the course; otherwise phase 1. Every step's result goes to S3 under the job's folder
(chosen by Studio), and a step whose result is already there is skipped, so a retried or restarted job
continues where it stopped. Progress goes to Studio after each step and every few seconds within one. If
the author starts over, Studio refuses the next report: the run stops there and deletes what it wrote since
Studio deleted the job's files.
"""

import datetime
import logging
import os
import tempfile
from pathlib import Path

from .. import audio
from ..llm import LLM, LLMError, LLMRefused
from ..storage import Storage
from . import clean, lessons, pieces, transcript, voices
from .monlam import Monlam, MonlamRefused, Transcriber
from .studio import JobGone, JobStopped, Progress, Studio, StudioUnavailable
from .suggest import SUGGESTION, NoTeacherText, OutlineProblem, Suggester

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.environ.get("MODEL_DIR", "/app/models"), "spkrec-ecapa-voxceleb")
SAMPLES_PER_VOICE = 3
SAMPLE_SECONDS = 20.0
RESPLIT_ROUNDS = 2  # how many times a garbled piece may be halved and resent
FIRST_SEND_SHARE = 0.9  # of the transcribing step's progress; the resend rounds share the rest
# Downloading and decoding the recording is the first part of whichever step a run starts with.
PREPARE_SHARE = 0.1
TRANSCRIBED = {"teacher"}

# Paths inside the job's folder; Studio keeps results relative to it.
ORIGINAL_FOLDER = "0-original/"
JOB_SUMMARY = "job.json"
TIMELINE = "1-voices/timeline.json"
WAVEFORM = "1-voices/waveform.json"
SAMPLE = "1-voices/samples/voice-{group}_sample-{number}.mp3"
CLEANED = "2-clean/clean.flac"
TRANSCRIPT_TXT = "3-transcript/transcript.txt"
TRANSCRIPT_JSON = "3-transcript/transcript.json"
RAW_TXT = "3-transcript/transcript-raw.txt"
RAW_JSON = "3-transcript/transcript-raw.json"
PLAN = "3-transcript/working/plan.json"
REPLIES = "3-transcript/working/results"
LESSON_CLIP = "5-lessons/lesson-{number}_{track}.mp3"
LESSONS_INDEX = "5-lessons/lessons.json"

RUNNABLE = ("queued", "finding_speakers", "cleaning", "transcribing", "suggesting", "cutting_audio", "writing_course")
CUT_SHARE = 0.95  # of the cutting step's progress; telling Studio to write the course is the rest

GENERIC_ERROR = "Something went wrong while processing the recording. Try again; if it stops again, ask for help."
MISSING_TRANSCRIPT = "The transcript is missing. Start over to transcribe the recording again."


class StepFailed(Exception):
    """A failure the author can act on; the message is shown on the tool page."""


def refusal_message(status: int) -> str:
    """Why Monlam refused, in words for the author."""
    if status in (401, 403):
        return "Monlam refused this server's key. Ask an administrator to check the Monlam key, then try again."
    if status == 402:
        return "The Monlam account has run out of credit. Once it is topped up, try again."
    return (
        f"Monlam refused part of the recording (error {status}). Try again; if it happens again, ask for help. "
        "The parts already done won't be sent (or paid for) again."
    )


def llm_message(error: LLMError) -> str:
    """Why the AI model couldn't write the course, in words for the author."""
    if isinstance(error, LLMRefused):
        if error.status in (401, 403):
            return (
                "OpenRouter refused this server's key. Ask an administrator to check the OpenRouter key, "
                "then try again."
            )
        if error.status == 402:
            return "The OpenRouter account has run out of credit. Once it is topped up, try again."
        return (
            f"The AI model refused to write the course (error {error.status}). Try again; if it happens again, "
            "ask for help."
        )
    return (
        "The AI model could not write the course just now. Try again in a few minutes; the parts already written "
        "are kept."
    )


def run(job_id: str, settings) -> None:
    """Entry point for the worker: one queue entry "course_from_recording:<job id>"."""
    studio = Studio(settings.cms_url, settings.service_token)
    try:
        job = studio.job(job_id)
    except JobGone:
        log.warning("Job %s no longer exists in Studio; nothing to do", job_id)
        return
    # A running status here means the service stopped halfway (e.g. a restart) and put the job back on
    # the queue: carry on with it. Anything else (waiting for the author, done, failed) is left alone.
    if job["status"] not in RUNNABLE:
        log.warning("Job %s is %s; nothing to do", job_id, job["status"])
        return
    storage = Storage(settings.s3_bucket)
    try:
        with tempfile.TemporaryDirectory(prefix="recording-") as tmp:
            RecordingJob(job, settings, studio, storage, Path(tmp)).run()
    except (JobStopped, JobGone) as stop:
        if isinstance(stop, JobStopped) and stop.status != "cancelled":
            log.warning("Job %s stopped: Studio refused its report (%s)", job_id, stop)
            return
        # The author started over while the job ran. Studio already deleted the job's files (and the
        # recording, unless kept), so remove what this run wrote since; the recording is Studio's to handle.
        deleted = storage.delete_folder(job["folder"], keep=ORIGINAL_FOLDER)
        log.info("Job %s: the author started over; stopped and deleted %s files written since", job_id, deleted)
    except StudioUnavailable:
        raise  # not the job's fault: the worker puts it back on the queue, and it continues from its last step
    except StepFailed as error:
        log.warning("Job %s failed: %s", job_id, error)
        studio.report(job_id, status="failed", error=str(error))
    except Exception:  # noqa: BLE001 - the author sees a plain message; the details are in the log
        log.exception("Job %s failed", job_id)
        studio.report(job_id, status="failed", error=GENERIC_ERROR)


class RecordingJob:
    def __init__(self, job: dict, settings, studio: Studio, storage: Storage, tmp: Path):
        self.job, self.settings, self.studio, self.storage, self.tmp = job, settings, studio, storage, tmp
        self.id, self.details = job["id"], job["details"]
        self.status = job["status"]

    def key(self, path: str) -> str:
        return f"{self.job['folder']}/{path}"

    def report(self, **fields) -> None:
        self.studio.report(self.id, **fields)

    def start_step(self, status: str) -> Progress:
        """Tell Studio the step has started; the first step of a run already did, while getting ready."""
        if status == self.status:
            return Progress(self.studio, self.id, start=PREPARE_SHARE)
        self.report(status=status, progress=0)
        self.status = status
        return Progress(self.studio, self.id)

    def first_step(self) -> str:
        if self.details.get("multiple_speakers") and not self.job["speaker_labels"]:
            return "finding_speakers"
        return "transcribing" if self.storage.exists(self.key(CLEANED)) else "cleaning"

    def check_can_finish(self) -> None:
        """Fail before any work (download, voices, the author's tags, cleaning) if transcription can't happen."""
        if self.details.get("teacher_language") != "bo":
            raise StepFailed("English transcription isn't available yet.")
        if not self.settings.monlam_api_key:
            raise StepFailed("Tibetan transcription isn't set up on this server yet (no Monlam key).")

    def phase(self) -> str:
        """What this run is for, from what is already done."""
        if self.job.get("course_approved") and self.job.get("course_draft"):
            return "build"
        if (self.job.get("results") or {}).get("transcript_json"):
            return "suggest"
        return "transcribe"

    def run(self) -> None:
        phase = self.phase()
        if phase == "suggest":
            self.suggest()
            return
        if phase == "build":
            self.build()
            return
        self.check_can_finish()
        # Say straight away which step this run is on: downloading a long recording takes a while,
        # and the author should see it move, not "waiting for its turn".
        self.status = self.first_step()
        self.report(status=self.status, progress=0)
        preparing = Progress(self.studio, self.id, end=PREPARE_SHARE * 0.8)
        original = self.storage.download(
            self.job["original_key"], self.tmp / f"original{Path(self.job['original_key']).suffix}", preparing
        )
        length = audio.duration(original)
        if length > self.settings.max_recording_minutes * 60:
            raise StepFailed(
                f"The recording is {round(length / 60)} minutes long; this tool takes recordings of up to "
                f"{int(self.settings.max_recording_minutes)} minutes."
            )
        self.report(duration_seconds=round(length, 1))
        self.write_summary(length)

        speech = audio.read(audio.to_wav(original, self.tmp / "work-16k.wav", voices.SR))
        Progress(self.studio, self.id)(PREPARE_SHARE)
        if self.details.get("multiple_speakers"):
            if not self.job["speaker_labels"]:
                self.find_voices(original, speech)
                return  # paused: the author tags the voices, and the job comes back on the queue
            timeline = self.storage.read_json(self.key(TIMELINE))
            if timeline is None:
                raise StepFailed("The voices that were found are missing. Upload the recording again.")
            turns = voices.name_turns(timeline, self.job["speaker_labels"])
        else:
            turns = voices.speech_turns(speech)

        pauses = pieces.find_pauses(speech)
        self.clean(original, turns, pauses)
        self.transcribe(speech, pauses, turns)

    def write_summary(self, length: float) -> None:
        """job.json: what this folder is, readable without the database."""
        self.storage.write_json(
            self.key(JOB_SUMMARY),
            {
                "job_id": self.id,
                "course_id": self.job["course_id"],
                "original_filename": self.job["original_filename"],
                "original_file": self.job["original_key"][len(self.job["folder"]) + 1 :],
                "minutes": round(length / 60, 1),
                "details": self.details,
                "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            },
        )

    # ---------- 1. find the voices (more than one speaker) ----------

    def find_voices(self, original: Path, speech) -> None:
        progress = self.start_step("finding_speakers")
        timeline = self.storage.read_json(self.key(TIMELINE))
        if timeline is None:
            try:
                timeline = voices.find_voices(speech, MODEL_DIR, self.settings.cpu_threads, progress)
            except voices.NoSpeech:
                raise StepFailed(
                    "No one could be heard speaking in this recording. Check that the right file was uploaded."
                )
            self.storage.write_json(self.key(TIMELINE), timeline)
        if not self.storage.exists(self.key(WAVEFORM)):
            self.storage.write_json(self.key(WAVEFORM), voices.waveform(speech))
        groups = []
        for voice in voices.voice_groups(timeline):
            samples = []
            spans = voices.sample_spans(voice["longest"], SAMPLES_PER_VOICE, SAMPLE_SECONDS)
            for number, (start, length) in enumerate(spans, 1):
                path = SAMPLE.format(group=voice["group"], number=number)
                clip = audio.cut_mp3(original, self.tmp / "sample.mp3", start, length)
                self.storage.upload(clip, self.key(path), "audio/mpeg")
                samples.append(path)
            groups.append({"group": voice["group"], "minutes": voice["minutes"], "samples": samples})
        log.info("Job %s: %s voices found; waiting for the author's tags", self.id, len(groups))
        self.report(
            status="needs_speaker_labels",
            progress=100,
            speaker_groups=groups,
            results={"timeline": TIMELINE, "waveform": WAVEFORM},
        )

    # ---------- 2. clean ----------

    def clean(self, original: Path, turns: list, pauses) -> None:
        if self.storage.exists(self.key(CLEANED)):
            return
        progress = self.start_step("cleaning")
        wav = audio.to_wav(original, self.tmp / "work-44k.wav", clean.SR)
        try:
            noise = clean.noise_power(clean.noise_sample(wav, clean.quiet_spans(turns, pauses)))
        except clean.NoStatic:
            # Not a moment of quiet to learn the static from: keep the sound as it is rather than stop.
            log.warning("Job %s: no quiet stretch to learn the static from; the audio is kept as it is", self.id)
            noise = None
        cleaned = clean.clean_file(wav, self.tmp / "clean.flac", noise, progress)
        wav.unlink()
        self.storage.upload(cleaned, self.key(CLEANED), "audio/flac")
        self.report(results={"clean": CLEANED})

    # ---------- 3. transcribe the teacher ----------

    def transcribe(self, speech, pauses, turns: list) -> None:
        progress = self.start_step("transcribing")

        plan = pieces.plan_pieces(speech, pauses, turns, TRANSCRIBED)
        if not plan:
            raise StepFailed("No teacher speech was found to transcribe.")
        transcriber = Transcriber(
            Monlam(self.settings.monlam_base_url, self.settings.monlam_api_key),
            self.storage,
            self.key(REPLIES),
            speech,
            key_prefix=self.id[:8],
        )
        minutes_before = float(self.job.get("transcribed_minutes") or 0)
        results, sent_total = {}, [0.0]

        def send(batch: list, start: float, end: float) -> None:
            """Transcribe ``batch``; its progress fills the step from ``start`` to ``end`` (shares of 0-1)."""
            batch_sent = [0.0]

            def on_progress(done: int, total: int, sent: float) -> None:
                batch_sent[0] = sent  # before the report, which stops the run if the author started over
                progress(start + (end - start) * done / max(1, total))

            try:
                failed = transcriber.send_all(batch, results, on_progress)
            except MonlamRefused as refused:
                raise StepFailed(refusal_message(refused.status)) from refused
            finally:
                # Recorded even when the run stops here: what was sent has been paid for.
                sent_total[0] += batch_sent[0]
                self.report(transcribed_minutes=round(minutes_before + sent_total[0] / 60, 2))
            if failed:
                raise StepFailed(
                    f"{len(failed)} parts of the recording could not be transcribed by Monlam. Try again: "
                    "the parts already done won't be sent (or paid for) again."
                )

        # Most of the step is the first sending; garbled pieces sent again in halves fill the rest.
        send(plan, 0.0, FIRST_SEND_SHARE)
        resend_share = (1 - FIRST_SEND_SHARE) / RESPLIT_ROUNDS
        for round_number in range(RESPLIT_ROUNDS):
            garbled = [
                p
                for p in plan
                if pieces.is_garbled(p, results[p["id"]].get("result") or {})
                and p["end"] - p["start"] >= 2 * pieces.MIN_HALF
            ]
            if not garbled:
                break
            log.info("Job %s: resending %s garbled pieces in halves", self.id, len(garbled))
            new_plan, halves = [], []
            for piece in plan:
                parts = pieces.halve(piece, pauses, speech) if piece in garbled else [piece]
                new_plan += parts
                halves += parts if piece in garbled else []
            plan = new_plan
            start = FIRST_SEND_SHARE + round_number * resend_share
            send(halves, start, start + resend_share)

        self.storage.write_json(self.key(PLAN), plan)
        lines = transcript.build_lines(plan, results, turns)
        self.storage.write_json(self.key(RAW_JSON), lines)
        self.storage.write_text(
            self.key(RAW_TXT), transcript.as_text(lines, "# What Monlam returned, before spelling fixes")
        )
        fixed = transcript.corrected(lines)
        self.storage.write_json(self.key(TRANSCRIPT_JSON), fixed)
        self.storage.write_text(
            self.key(TRANSCRIPT_TXT), transcript.as_text(fixed, "# Transcript, Tibetan spelling fixed")
        )
        self.report(
            status="transcript_ready",
            progress=100,
            results={"transcript": TRANSCRIPT_TXT, "transcript_json": TRANSCRIPT_JSON, "transcript_raw": RAW_TXT},
        )

    # ---------- phase 2, step 1: suggest a course ----------

    def suggest(self) -> None:
        if not self.settings.openrouter_api_key:
            raise StepFailed("Writing the course isn't set up on this server yet (no OpenRouter key).")
        self.status = "suggesting"
        self.report(status="suggesting", progress=0)
        lines = self.storage.read_json(self.key(TRANSCRIPT_JSON))
        if lines is None:
            raise StepFailed(MISSING_TRANSCRIPT)
        suggester = Suggester(
            LLM(self.settings.openrouter_api_key, self.settings.openrouter_model), self.storage, self.key, self.details
        )
        try:
            draft = suggester.run(lines, self.job.get("transcript_edits") or {}, Progress(self.studio, self.id))
        except NoTeacherText:
            raise StepFailed("The transcript has none of the teacher's words to make a course from.")
        except OutlineProblem as problem:
            log.warning("Job %s: the model's outline didn't fit the transcript: %s", self.id, problem.args[0])
            raise StepFailed(
                "The suggested outline of the course didn't fit the transcript, even when asked again. Try again."
            )
        except LLMError as error:
            log.warning("Job %s: %s", self.id, error)
            raise StepFailed(llm_message(error)) from error
        self.report(status="course_draft_ready", progress=100, course_draft=draft,
                    results={"course_suggestion": SUGGESTION})

    # ---------- phase 2, step 2: cut the lesson audio, then Studio writes the course ----------

    def build(self) -> None:
        # Each lesson's clips are named by its place in the draft ("1-2": section 1, lesson 2).
        kept = [
            (f"{s_number}-{l_number}", lesson)
            for s_number, section in enumerate(self.job["course_draft"]["sections"], start=1)
            for l_number, lesson in enumerate(section["lessons"], start=1)
            if lesson.get("include", True)
        ]
        if not kept:
            raise StepFailed("No lessons were kept, so there is nothing to add to the course.")

        # Lessons already cut (lessons.json, written after each one) are kept, so a retry continues. One without
        # its Tibetan track is cut again (an earlier version left out short bits of the teacher).
        done = self.storage.read_json(self.key(LESSONS_INDEX)) or {}
        todo = [
            (number, lesson)
            for number, lesson in kept
            if lesson["id"] not in done
            or not done[lesson["id"]]["tracks"].get("bo")
            or not all(self.storage.exists(self.key(path)) for path in done[lesson["id"]]["tracks"].values())
        ]
        # Nothing left to cut (e.g. this run comes after Studio said it was still writing the course): straight
        # on to writing, so the author doesn't see the step go back to cutting.
        if todo:
            self.status = "cutting_audio"
            self.report(status="cutting_audio", progress=0)
            lines = self.storage.read_json(self.key(TRANSCRIPT_JSON))
            if lines is None:
                raise StepFailed(MISSING_TRANSCRIPT)
            self.cut_lessons(todo, lines, done)

        # Studio needs every lesson's Tibetan track. Cutting keeps even short bits of the teacher, so only a
        # lesson with no sound of the teacher at all lacks one.
        silent = [lesson["title"] for _, lesson in kept if not done[lesson["id"]]["tracks"].get("bo")]
        if silent:
            raise StepFailed(
                f"There is no Tibetan audio to cut for the lesson “{silent[0]}”. Start over, and check the voices "
                "are tagged right."
            )

        self.status = "writing_course"
        self.report(status="writing_course", progress=int(CUT_SHARE * 100), results={"lessons": LESSONS_INDEX})
        answer = self.studio.write_course(
            self.id, {lesson["id"]: {"tracks": done[lesson["id"]]["tracks"]} for _, lesson in kept}
        )
        # Studio sets the job's status itself: course_ready, or failed with a message for the author.
        log.info("Job %s: Studio wrote the course: %s", self.id, answer.get("status"))

    def cut_lessons(self, todo: list, lines: list, done: dict) -> None:
        # Before the long download of the original.
        if not self.storage.exists(self.key(CLEANED)):
            raise StepFailed("The cleaned recording is missing. Start over to process the recording again.")
        preparing = Progress(self.studio, self.id, end=PREPARE_SHARE * 0.8)
        original = self.storage.download(
            self.job["original_key"], self.tmp / f"original{Path(self.job['original_key']).suffix}", preparing
        )
        speech = audio.to_wav(original, self.tmp / "work-16k.wav", pieces.SR)
        original.unlink()
        cleaned = self.storage.download(self.key(CLEANED), self.tmp / "clean.flac")
        progress = Progress(self.studio, self.id, start=PREPARE_SHARE, end=CUT_SHARE)
        progress(0)

        turns = lessons.turns_from_lines(lines)
        for count, (number, lesson) in enumerate(todo, start=1):
            tracks = lessons.cut_lesson(cleaned, speech, clean.SR, turns, lesson["start"], lesson["end"])
            entry = {"number": number, "tracks": {}, "minutes": {}, "pieces": {}}
            for track, cut in tracks.items():
                path = LESSON_CLIP.format(number=number, track=track)
                mp3 = lessons.write_mp3(cut["sound"], clean.SR, self.tmp / f"lesson-{number}_{track}.mp3")
                self.storage.upload(mp3, self.key(path), "audio/mpeg")
                mp3.unlink()
                entry["tracks"][track] = path
                entry["minutes"][track] = round(len(cut["sound"]) / clean.SR / 60, 1)
                # Where each piece comes from in the recording, to check a lesson by ear.
                entry["pieces"][track] = [
                    [round(piece["start"], 2), round(piece["end"], 2), piece["voices"]] for piece in cut["pieces"]
                ]
            done[lesson["id"]] = entry
            self.storage.write_json(self.key(LESSONS_INDEX), done)
            progress(count / len(todo))
        log.info("Job %s: cut the audio of %s lessons", self.id, len(todo))
