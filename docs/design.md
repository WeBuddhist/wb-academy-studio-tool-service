# Course from a recording — platform design

How the audio pipeline (prototyped as scripts in `dtbw-audio-processing-pipeline`) runs inside the
WeBuddhist Open edX platform, so a course author can upload a teaching recording in Studio and get a
course draft back. This repo is the service that does the heavy processing.

Status (2026-11-06): **phase 1 is built** (recording → checked transcript). **Phase 2** (transcript →
draft course in Studio) is designed in [Phase 2: transcript to draft course](#phase-2-transcript-to-draft-course)
and not built yet.

## Decided

| Topic | Decision |
|---|---|
| Storage | All files (recording, cleaned audio, transcripts, lesson clips) live in **S3**, in a private bucket. The finished course's lesson audio is copied to the **public** assets bucket, so the course can play it (phase 2). Nothing goes into Studio's Files & Uploads. |
| Processing | Runs **in the background**, never inside a web request. The page shows progress and can be closed and reopened. |
| Where it runs | A **separate audio service**: its own Docker container, separate from the LMS and CMS. |
| Hosting | Production is `tutor local` on one server, shared with Open edX. The service is one more container there. |
| Queue | **Redis**, which `tutor local` already runs. No new AWS service. |
| Limits | Service container: `mem_limit: 4g`, `cpus: 2`, **one job at a time**. |
| Entry point | Studio empty course outline → "Create with a tool" → "From a teaching recording". |
| Frontend | `frontend-app-studio-extension`, route `/course-from-recording/:courseId`. One route per tool; the tool's steps are screens inside that page. |
| Styles | Only class names in the MFE; all styling in `brand-openedx/paragon/_studio-extension.scss`. |
| Speakers | The author says whether there is **more than one speaker**. One speaker: no speaker detection, the whole recording is transcribed. More than one (e.g. teacher and translator): **speaker detection is the first step**, the author tags the voices, and **only the teacher is transcribed**. |
| Order of the steps | More than one speaker: find the voices → author tags them → clean → transcribe the teacher. One speaker: clean → transcribe. |
| Tagging the voices | With more than one speaker, the service splits the voices into groups by itself (`detect_speakers.py --groups`, 3 by default: teacher, translator and anything else such as chanting). It pauses the job at `needs_speaker_labels` with a few short sample clips per group; the author listens and tags each group (teacher, translator, chanting, other), and the job continues. Several groups may be the teacher, since one voice is sometimes split in two. |
| Languages | The author declares the **teacher's language**. Tibetan is transcribed with **Monlam AI**, English with **Gemini** (same key and model as the AI course creator). Any other language is refused at upload as not supported yet. The translator's language is recorded but not transcribed. |

### Already done

- Studio "Create with a tool" menu and plugin slot (`frontend-app-authoring`), plus brand styles.
- Tutor wiring for the studio-extension URL and the menu item (`tutor-plugins/extension_mfe.py`).
- studio-extension route `/course-from-recording/:courseId` with a placeholder page.

## The parts

| Part | Repo | Job |
|---|---|---|
| **Tool page** | `frontend-app-studio-extension` | Upload, progress and review screens. |
| **Plugin app** `course_from_recording` | `sherab-custom-plugin` (CMS) | Checks the user can author the course, keeps the job and its status in MySQL, hands out S3 upload links, puts jobs on the queue, writes the final course. |
| **Audio service** | `wb-academy-studio-tool-service` (this repo) | Takes a job from the queue, runs the pipeline steps, saves outputs to S3, reports status to the plugin. Knows nothing about courses. |
| **S3** | AWS | Stores every file. |
| **Redis** | Tutor | The job queue. |
| **Tutor plugin** | `tutor-plugins/` | Adds the service container, its limits and its settings to `docker-compose`. |

Why a separate service and not a Celery task in `cms-worker`:

- PyTorch and SpeechBrain (a few GB) stay out of the Open edX images.
- If audio processing runs out of memory, only the service container is killed, never Studio, the LMS or MySQL.
- The pipeline scripts are reused as they are, without following edx-platform's dependency versions.
- The service can be stopped or given more resources without touching Open edX.

## The flow

```
Tool page (browser)          Plugin (CMS)                   Audio service
───────────────────          ────────────                   ─────────────
1  POST /jobs/  ───────────► create job (status: uploading)
                ◄─────────── job id + presigned S3 upload URL
2  PUT file ─────────────────────────────────────────────► S3
3  POST /jobs/<id>/start/ ─► check file is in S3
                             status: queued, push id ──────► Redis queue
                                                             4  take job id
                                                                download from S3
                                                                run steps one by one
                             ◄──── POST status after each step (signed)
                                                                upload outputs to S3
5  GET /jobs/<id>/ every few seconds → shows the current step
6  POST /jobs/<id>/suggest/ ► queue again ─────────────────► suggest an outline (model via OpenRouter)
                             ◄──── course draft
7  PUT /jobs/<id>/course-draft/ (author shapes it)
8  POST /jobs/<id>/build/ ───► queue again ─────────────────► cut the lesson audio
                             ◄──── POST service/jobs/<id>/course/
                             copy the clips to the public bucket,
                             write sections, units, text and questions into the course (unpublished)
```

Steps 6 to 8 are phase 2; see [Phase 2](#phase-2-transcript-to-draft-course).

The browser uploads directly to S3 with a presigned URL, so the large file never goes through the CMS.

## Job

One Django model in the plugin app, `RecordingJob`:

| Field | Notes |
|---|---|
| `id` | UUID; also the S3 folder name. |
| `course_id`, `user` | Who started it, and for which course. |
| `status` | See the statuses below. |
| `progress` | 0–100 for the current step. |
| `error` | Message shown to the author when a step fails. |
| `metadata` | JSON: teacher, `multiple_speakers`, `teacher_language` (`bo` or `en`), `translator_language`, text, place, year. |
| `original_filename`, `duration_seconds` | `duration_seconds` is filled by the service. |
| `transcribed_minutes` | Minutes sent for transcription (Monlam or Gemini), for cost tracking. |
| `results` | JSON: S3 keys of each step's outputs. |
| `created`, `modified` | |

Statuses, in order:

```
More than one speaker:
uploading → queued → finding_speakers → needs_speaker_labels → queued → cleaning → transcribing → transcript_ready
One speaker:
uploading → queued → cleaning → transcribing → transcript_ready                        (phase 1 stops here)
Then (phase 2):
transcript_ready → queued → suggesting → course_draft_ready
                 → queued → cutting_audio → writing_course → course_ready
any step  → failed
```

Speaker detection is the first step, and only for recordings with more than one speaker. At
`needs_speaker_labels` the job is off the queue and waits for the author; tagging the voices puts it back
on the queue, and the service continues with cleaning and transcription.

The author can start over from any step: Studio marks the job `cancelled` and deletes its files (and the
recording, unless the author keeps it to reuse). The service finds out from its next progress report, which
Studio refuses with `409` and `"status": "cancelled"`: it stops there, drops the pieces not yet sent to
Monlam, still records the minutes already sent (Studio accepts that one report), and deletes anything it
wrote in the job's folder since, except `0-original/`, which is Studio's to handle.

The tool page looks up the latest job for the course and opens the screen for its status, so the author
can leave and come back.

## Plugin API

The `course_from_recording` app in `sherab-custom-plugin`, mounted on the Studio host. Full details
are in that repo's README.

Author endpoints: JWT or session login, plus `has_course_author_access` for the job's course.

| Method and path | Purpose |
|---|---|
| `POST /api/course-from-recording/v1/jobs/` | Create a job (`course_id`, `filename`, `content_type`, `size`, `consent`, `details`); returns the job and a signed S3 upload link. Type and size are part of the signature. |
| `GET /api/course-from-recording/v1/jobs/?course_id=…` | Latest job for a course, used to resume. |
| `GET /api/course-from-recording/v1/jobs/<id>/` | Status, progress, error and short-lived download links to the outputs. |
| `POST /api/course-from-recording/v1/jobs/<id>/start/` | Check the upload is in S3 with the declared size, then queue the job. |
| `POST /api/course-from-recording/v1/jobs/<id>/speakers/` | The author's tag for each voice group (`teacher`, `translator`, `chanting`, `other`; at least one teacher), then queue the job again. |
| `POST /api/course-from-recording/v1/jobs/<id>/retry/` | Queue a failed job again; it continues from the failed step. |
| `POST /api/course-from-recording/v1/jobs/<id>/start-over/` | Stop the job and delete its files; the recording too unless `keep_recording`. |
| `POST /api/course-from-recording/v1/jobs/<id>/reuse-recording/` | A new job with a kept recording (moved in S3, not uploaded again), queued straight away. |
| *phase 2* `POST …/v1/jobs/<id>/suggest/` | Only from `transcript_ready`: queue the job to suggest a course outline. |
| *phase 2* `PUT …/v1/jobs/<id>/course-draft/` | Only at `course_draft_ready`: save the author's changes to the draft (titles, lessons switched off, summaries, key points, questions). Lesson ids and line ranges can't change. |
| *phase 2* `POST …/v1/jobs/<id>/build/` | Only at `course_draft_ready`, with `approved: true` and at least one lesson kept: queue the job to cut the audio and write the course. |

Service endpoints: no user login; every request needs the shared secret in the `X-Service-Token` header.

| Method and path | Purpose |
|---|---|
| `GET /api/course-from-recording/v1/service/jobs/<id>/` | Job details, its S3 folder, the recording's key, the speaker groups and the author's tags. The service reads these instead of trusting anything in the queue entry. |
| `PATCH /api/course-from-recording/v1/service/jobs/<id>/` | Progress report: any of `status`, `progress`, `error`, `results` (merged), `duration_seconds`, `transcribed_minutes`, `speaker_groups`, and in phase 2 `course_draft` (with `course_draft_ready`). Only while the job is queued or running. |
| *phase 2* `POST /api/course-from-recording/v1/service/jobs/<id>/course/` | Only at `writing_course`: the lesson clips the service cut (each lesson needs its `bo` track). Studio copies them to the public bucket and writes the course. Safe to call twice (a written course is returned as it is; one still being written answers `503`). |

## Audio service

The Python package `studio_tool_service`, run in its own container. Each Studio tool gets its own
sub-package, so later tools can be added next to it:

```
src/studio_tool_service/
    worker.py                  main loop: take a job from Redis, run it, repeat
    config.py                  settings from environment variables
    storage.py                 S3: download, upload, read and write JSON
    audio.py                   ffmpeg/soundfile: decode to a mono WAV working copy, read parts, cut clips
    course_from_recording/
        job.py                 the steps in order; skips steps whose result is already in S3; reports to Studio
        studio.py              the sherab plugin's service API (job details, progress reports)
        voices.py              find the voices (from detect_speakers.py), or the quiet parts for one speaker
        clean.py               remove the static, block by block (from denoise.py)
        pieces.py              cut the teacher's turns into pieces; check replies (from transcribe.py)
        monlam.py              send pieces to Monlam, saving every reply to S3 (from transcribe.py)
        transcript.py          build the transcript; Tibetan spelling rules (from correct_transcript.py)
Dockerfile
```

The Tutor plugin that adds the container, its limits and its settings (`studio_tool_service.py`) lives
in the Tutor plugins folder with the other WeBuddhist plugins, not in this repo.

The step code is copied from the pipeline repo's scripts and turned into functions that take input and
output paths, without the command-line and `output/` folder logic.

The main loop:

1. Wait for a job on the Redis list `studio_tool_service:jobs` (Redis database 3). Each entry is
   `<tool>:<job id>`, e.g. `course_from_recording:3f2c…`, so later tools share the same queue. `BLMOVE`
   moves it to `studio_tool_service:jobs:processing`, so a job is never lost if the container restarts.
   On start-up, anything left in `processing` goes back to the front of the queue.
2. Download the recording from S3 into a temporary folder.
3. Run the steps in order. A step whose output is already in S3 is skipped, so a retried or restarted
   job continues where it stopped.
4. After each step: upload its outputs and `POST` the new status to the plugin.
5. Delete the temporary folder and take the next job.

| Step | Copied from (pipeline repo) | Output in S3 |
|---|---|---|
| Read duration, check the limit | — | `duration_seconds` on the job |
| Find speakers (only with more than one speaker), then pause for the author's tags | `detect_speakers.py` | `1-voices/timeline.json`, `1-voices/samples/` clips |
| Clean | `denoise.py` | `2-clean/clean.flac` |
| Transcribe the teacher: Tibetan with Monlam, English with Gemini | `transcribe.py` (Monlam); new for Gemini | `3-transcript/transcript-raw.txt` |
| Correct (Tibetan) | `correct_transcript.py` (rules and glossary part) | `3-transcript/transcript.txt`, `transcript.json` |

Cleaning learns what the static sounds like from the stretches where nobody speaks. With more than one
speaker those come from speaker detection (`segments.json`, "no speech"). With one speaker, speaker
detection is skipped, so the clean step finds the quiet stretches itself with the same loudness check
that speaker detection starts with (`speech_mask` in `detect_speakers.py`), without sorting voices.
Transcription uses the original recording; the cleaned audio is for the lesson clips.

Gemini is a general model, not a speech-to-text service, so the English step asks it for a word-for-word
transcript (no summarising or tidying) and sends the audio in pieces of about 20 minutes: one reply has
a length limit, and each piece starts at a known time, which keeps the timestamps usable for cutting
lesson audio later.
| *phase 2* Suggest the outline, then each lesson's summary, key points and questions | new, the model through OpenRouter | `4-course/suggestion.json`, `4-course/working/` |
| *phase 2* Cut the lesson audio (Tibetan and translator tracks) | `cut_unit_audio.py` | `5-lessons/lesson-<s>-<n>_bo.mp3`, `…_translator.mp3`, `5-lessons/lessons.json` |

### S3 layout

```
<private Open edX bucket>/studio-tools/
└── course-from-recording/
    └── Sherab-tspc12-2026/                        one folder per course
        └── 2026-11-03_audio-tibetan_a101e248/      one per recording: date, file name, short job id
            ├── job.json                            summary: teacher, language, speakers, file name, dates
            ├── 0-original/
            │   └── audio-tibetan.mp3               the upload, never changed
            ├── 1-voices/                           only with more than one speaker
            │   ├── timeline.json                   who speaks when (teacher, translator, chanting…)
            │   └── samples/voice-1_sample-1.mp3 …  short clips for the author to tag the voices
            ├── 2-clean/
            │   └── clean.flac                      the recording with the static removed
            ├── 3-transcript/
            │   ├── transcript.txt                  readable, spelling fixed
            │   ├── transcript.json                 the same, with times, for the next steps
            │   ├── transcript-raw.txt              what Monlam or Gemini returned, before fixes
            │   └── working/                        pieces sent, raw replies, progress log
            ├── 4-course/                           phase 2
            │   ├── suggestion.json                 the model's outline and lessons, never changed
            │   └── working/                        each model reply, so a retry redoes only what's missing
            └── 5-lessons/                          phase 2
                ├── lesson-1-1_bo.mp3               Tibetan track (teacher and chanting)
                ├── lesson-1-1_translator.mp3       translator track
                └── lessons.json                    each lesson's files, length and joins
```

Studio chooses the folder once, when the job is created, and saves it on the job; the service reads it
from Studio (`folder`, `original_key`) and never builds paths itself. Names are lowercase ASCII with
dashes, so links and command-line tools need no escaping; the course id loses its `course-v1:` and
`+` (`course-v1:Sherab+tspc12+2026` → `Sherab-tspc12-2026`), and a file name with nothing left after
that (e.g. all Tibetan) becomes `recording`. The real file name stays in the database and `job.json`.

The bucket is private. The browser only gets presigned URLs, which expire.

The finished course's audio goes to the **public assets bucket** (`sherab-public-assets-dev` for local
and staging, `sherab-public-assets` for production; the same bucket as partner logos and profile images,
`S3_PARTNER_LOGO_BUCKET`). Its policy lets anyone read any file, so the course's players need no login and
the links never expire:

```
<public assets bucket>/course-audio/
└── Sherab-tspc12-2026/
    └── 2026-11-03_audio-tibetan_a101e248/          same name as the job folder
        ├── lesson-1-1_bo.mp3
        └── lesson-1-1_translator.mp3
```

These files belong to the course: Start over never deletes them.

## Secrets and settings

All values are set in Tutor config on the server, never in git. The `studio_tool_service` Tutor plugin
defines them; the full list and the setup steps are in [tutor-setup.md](tutor-setup.md).

| Setting | Used by |
|---|---|
| `STUDIO_TOOL_SERVICE_MONLAM_API_KEY` | Service only. |
| `STUDIO_TOOL_SERVICE_OPENROUTER_API_KEY` / `..._OPENROUTER_MODEL` (default: Tutor's `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`, the AI course creator's) | Service only. |
| `STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID` / `..._SECRET_ACCESS_KEY`; in production a key limited to `studio-tools/*` | Service. The plugin uses the Open edX key. |
| `STUDIO_TOOL_SERVICE_BUCKET` (default: the private Open edX bucket) and `..._PREFIX` (`studio-tools`) | Service and plugin. |
| `STUDIO_TOOL_SERVICE_TOKEN` (shared secret for the status API, generated by Tutor) | Service and plugin. |
| `STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES` (longest recording accepted) | Plugin and service. |
| `STUDIO_TOOL_SERVICE_REDIS_URL` (Tutor's Redis, database 3) and `..._QUEUE` | Service and plugin. |
| *phase 2* `STUDIO_TOOL_SERVICE_PUBLIC_BUCKET` / `..._PUBLIC_DOMAIN` (default: `S3_PARTNER_LOGO_BUCKET` / `S3_PARTNER_LOGO_CUSTOM_DOMAIN`) | Plugin only: it copies the lesson audio there with the Open edX key. |

The plugin adds the container through the `local-docker-compose-services` patch (and
`local-docker-compose-dev-services` for `tutor dev`), and gives Studio one `STUDIO_TOOL_SERVICE`
settings dict through `openedx-cms-common-settings`.

## Cost control

- Monlam costs about **$0.032 per minute** sent (about $2 per hour). Only the teacher is sent.
- English goes to Gemini, billed on the same Google account as the AI course creator.
- Phase 2's outline and lesson writing use Gemini through OpenRouter, billed to the same OpenRouter account
  as the AI course creator: one call for the outline and one per lesson, text only, so a few cents per
  recording.
- A recording longer than `STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES` is rejected before transcription.
- Minutes sent are stored on each job (`transcribed_minutes`), so monthly spend can be added up.

## Phases

**Phase 1 — recording to transcript.** Proves the whole chain: upload, queue, service, S3, status and resume.

- Plugin: model, API, presigned upload, queue push, status endpoint.
- Service: container, loop, steps: clean, find speakers, transcribe, correct.
- Tutor plugin: service container with limits; settings.
- Tool page: upload screen (with the speakers and language questions), progress screen, "tag the voices" screen, transcript screen with downloads.

**Phase 2 — transcript to draft course** (designed below; the earlier "phase 3", writing into the
course, is part of it).

- Service: suggest the outline and lessons with Gemini; cut the lesson audio.
- Plugin: draft, approval and build endpoints; copy the audio to the public bucket; write the course.
- Tool page: "Suggest a course" on the transcript screen, the Shape your course screen, the Draft course
  ready screen.

**Later:** listen before and after cleaning, choose which parts to keep, email when a job finishes,
"who can receive this teaching", synced transcript (text following the audio), switching tracks with
`xblock-multilingual-text`, Open Response reflections, practice plans, Tibetan translations of the tool's
labels.

## Open decisions

1. **Two-file recordings:** the current pipeline also handles a teacher and translator recorded
   separately. Phase 1 assumes one recording per job.

Decided since the first draft: the AI model is **Gemini**; lesson audio is a plain `<audio>` player in a
Text component, with the files in the **public assets bucket**.

## Phase 2: transcript to draft course

Follows the design files' step 6 ("Shape your course") and step 7 ("Draft course ready").

### Decided

| Topic | Decision |
|---|---|
| AI model | **Gemini through OpenRouter** (openrouter.ai: one OpenAI-style API for many models), called **from the service** with its own small client (`studio_tool_service/llm.py`, plain `requests`, like Monlam, so the image needs no new library), using the service's `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` (default: the AI course creator's key and model, `google/gemini-2.5-flash-lite`). Changing the model is a setting. Not through Studio: each call takes 10–30 s and would hold one of Studio's few web workers, and the background service already gives progress, resume and Start over. |
| Language | The course is written in **Tibetan only** for now. |
| Levels | Normal Open edX levels: section → subsection → unit. **Gemini decides how many** sections and lessons, from the teaching's natural topic breaks. |
| A lesson | One **subsection**, split into four unit pages: **Listen**, **Key points**, **Check**, **Reflect**. No "full recording" unit. |
| Components | Only **Text** and **Problem**. Multiple choice is ungraded practice with no try limit of its own (the course's "Maximum Attempts" setting applies: unlimited while it is empty); reflection questions are plain text, nothing is submitted. |
| Audio | A **Tibetan track** (teacher and chanting) and the **translator's track**, cut at real pauses (as `cut_unit_audio.py` does), MP3 mono 64 kbps, played with `<audio>` from the public assets bucket. |
| Author's review | The Shape your course screen: rename sections and lessons, switch lessons off, edit summaries, key points and questions. Lesson breaks can't be moved. A required tick: "The teacher or their office has reviewed this outline." |
| Publishing | The tool **never publishes**. The course is written as an unpublished draft at the end of the course outline; authors keep editing it there, by hand or with the AI course creator's "Edit with SherabAI" section editor, which works on any section. |
| Not now | "Who can receive this teaching", the email when a job finishes, synced transcript, the multilingual block, Open Response. |

### Who does what

- **The service** does the slow work: the Gemini calls and cutting the audio. It knows nothing about
  courses.
- **Studio** writes the course, since only Studio can write course content. It also copies the clips to
  the public bucket, with the Open edX key that already writes logos there, so the service's key stays
  limited to `studio-tools/*`.
- The service works out what to run from what is already done, as it resumes today: an approved draft →
  build; a transcript but no draft → suggest; otherwise phase 1.

### Suggesting the course (service, status `suggesting`)

1. Read `3-transcript/transcript.json` and lay the author's corrections (`transcript_edits`) over it.
2. **Outline:** one call to the model with the recording's details and the teacher's lines, numbered and
   timed. It returns sections and lessons, each lesson a range of line numbers, with Tibetan titles. The
   service checks the ranges (in order, no overlaps, nothing outside the transcript; a small gap joins the
   lesson before it) and Studio's limits (at most 30 sections, 50 lessons in a section), and asks once more
   with the problems if they are wrong.
   **Each passage stays with its translation.** With a translator, the teacher says a passage and the
   translator then translates it. A lesson may only start where the teacher speaks again after the
   translator, so no lesson ends with Tibetan that is translated in the next one. The lines the model reads
   are marked "(translator)" where the translator speaks. A break it still puts mid-passage is moved back to
   the passage's start, and a lesson that lies within one passage joins the lesson before it.
3. **Lessons:** one call per lesson, three at a time, for its summary, 3–5 key points, 2–3 multiple-choice
   questions (3–4 options, one right answer, a short explanation) and 3 reflection questions. A reply with
   English (Latin-script) words in it is asked for once more. Each reply is saved in `4-course/working/`,
   so a retry only redoes the lessons still missing.
4. Save `4-course/suggestion.json` and report `course_draft_ready` with the draft.

Progress: the outline is the first 30 %, the lessons share the rest. Prompts are files in
`src/studio_tool_service/course_from_recording/prompts/`, and replies use strict JSON Schema (`response_format`), with OpenRouter routing only to providers
that support it. A request that timed out is sent again only once: the provider may still be writing (and
billing) the first answer.

### The draft

Studio keeps the editable copy on the job (`course_draft`); the suggestion in S3 never changes.

```json
{"version": 1, "course_title": "…",
 "sections": [{"id": "s1", "title": "…",
   "lessons": [{"id": "s1-l1", "title": "…", "include": true,
     "first_line": 12, "last_line": 48, "start": 123.4, "end": 890.2, "minutes": 12.8,
     "summary": "…", "key_points": ["…"],
     "mcq": [{"question": "…", "options": ["…", "…", "…"], "answer": 0, "explanation": "…"}],
     "reflection": ["…", "…", "…"]}]}]}
```

- `first_line` / `last_line` are line numbers in `transcript.json` (the same numbers as `transcript_edits`).
- `start` is the lesson's first teacher line; `end` is the next lesson's start (or the last speech), so the
  translator's reply to the last line stays in the lesson. The first lesson starts with the first speech of
  the recording, so opening prayers and chanting are in it.
- The Listen page's text is not in the draft: it is the teacher's own lines in the range, with the
  author's corrections, one paragraph per turn.

### Building the course (status `cutting_audio`, then `writing_course`)

1. **Cut the audio** (service). For each lesson kept, from the cleaned recording:
   - the teacher and chanting go to the Tibetan track, the translator to the translator track, and
     "other" and silence are left out;
   - each switch is moved to the longest real pause within 2 s;
   - pieces closer than 0.5 s are merged and pieces under 1.5 s dropped, unless the teacher has nothing
     longer in the lesson: every lesson keeps its Tibetan track, which Studio requires;
   - the rest are joined with 0.6 s of silence and short fades.

   Clips already in `5-lessons/` are kept, so a retry continues (and when everything is cut, the run goes
   straight to `writing_course`). A translator track with nothing in it gets no file; a lesson with no sound
   of the teacher at all fails the job, naming the lesson.
2. **Write the course** (Studio, `POST service/jobs/<id>/course/`):
   - copy the clips to `course-audio/…` in the public bucket;
   - build the course with `ai_course_creator/services/course_builder.py` (`apply_course_json`), which
     appends the sections at the end of the outline, never publishes, and removes what it made if
     anything fails;
   - save the created sections on the job, and finish at `course_ready`.

   Writing runs outside a database transaction, one request at a time: while a course is still being
   written, Studio answers `503`, and the service asks again later. If the person who started the job can no
   longer edit the course, it fails with that message.

**Reused from the AI course creator** (`sherab-custom-plugin/ai_course_creator/services/course_builder.py`):

- `apply_course_json`: the approved draft is turned into its course JSON and written by it unchanged.
- `build_problem_olx`: the multiple-choice questions.
- `user_can_author`, and its rollback through a `strict` option: a page that can't be made removes
  everything written.

Small options are added there, both off by default so the AI course creator is unchanged:

- `maxAttempts` on a problem (1 by default; the recording tool's practice questions set none of their own, so
  the course's "Maximum Attempts" setting applies);
- a label for the answer explanation's heading (it is always "Explanation" in English today).

What each lesson looks like in Studio:

```
Subsection: <lesson title>
├── Unit: Listen      Text: the Tibetan player, the translator's player (labelled with the
│                     translator's language), then the teacher's words
├── Unit: Key points  Text: the summary and the key points
├── Unit: Check       Problem: one multiple-choice question per component, ungraded
└── Unit: Reflect     Text: the reflection questions
```

The Tibetan names of the four pages are kept in one place in the plugin, to be checked by a Tibetan
reader.

### Tool page

- **Transcript screen:** a "Suggest a course" button.
- **Progress screen:** "Suggesting a course" (reading the whole teaching, finding natural topic breaks,
  drafting summaries and questions) and "Building your draft course" (cutting the lesson audio, adding
  it to Studio as a draft).
- **Shape your course:**
  - each section with an editable title, and its lessons, each with an on/off switch, an editable title,
    its length, and Preview/Edit for the summary, key points and questions;
  - changes save as the author goes;
  - the footer shows "N lessons · M min of audio", the review tick and "Create draft course".
- **Draft course ready:**
  - "Nothing is published yet…";
  - the number of sections and lessons, and the minutes of audio;
  - the outline as written;
  - "Open in course outline".
- Start over stays available. Once the course is written, starting over leaves the course and its audio
  in place.
