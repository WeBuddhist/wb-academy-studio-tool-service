# Course from a recording — platform design

How the audio pipeline (prototyped as scripts in `dtbw-audio-processing-pipeline`) runs inside the
WeBuddhist Open edX platform, so a course author can upload a teaching recording in Studio and get a
course draft back. This repo is the service that does the heavy processing.

Status: **draft for discussion** (2026-11-03). Nothing below the "Already done" line is built yet.

## Decided

| Topic | Decision |
|---|---|
| Storage | All files (recording, cleaned audio, transcripts, lesson clips) live in **S3**, in a private bucket. |
| Processing | Runs **in the background**, never inside a web request. The page shows progress and can be closed and reopened. |
| Where it runs | A **separate audio service**: its own Docker container, separate from the LMS and CMS. |
| Hosting | Production is `tutor local` on one server (8 cores, 31 GB RAM, no swap). The service is one more container there. |
| Queue | **Redis**, which `tutor local` already runs. No new AWS service. |
| Limits | Service container: `mem_limit: 4g`, `cpus: 2`, **one job at a time**. |
| Entry point | Studio empty course outline → "Create with a tool" → "From a teaching recording". |
| Frontend | `frontend-app-studio-extension`, route `/course-from-recording/:courseId`. One route per tool; the tool's steps are screens inside that page. |
| Styles | Only class names in the MFE; all styling in `brand-openedx/paragon/_studio-extension.scss`. |

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
- If audio processing runs out of memory, only the service container is killed, never Studio, the LMS or MySQL. This matters because the server has no swap.
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
6  Author reviews the draft (later phase)
7  POST /jobs/<id>/publish/ ► write sections, units, text, questions and audio into the course
```

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
| `metadata` | JSON: teacher, text, languages, place, year, consent. |
| `original_filename`, `duration_seconds` | `duration_seconds` is filled by the service. |
| `transcribed_minutes` | Minutes sent to Monlam, for cost tracking. |
| `results` | JSON: S3 keys of each step's outputs. |
| `created`, `modified` | |

Statuses, in order:

```
uploading → queued → cleaning → finding_speakers → transcribing → transcript_ready      (phase 1 stops here)
          → structuring → writing → cutting_audio → ready_for_review → publishing → done (later phases)
any step  → failed
```

The tool page looks up the latest job for the course and opens the screen for its status, so the author
can leave and come back.

## Plugin API

Mounted on the Studio host like the AI course creator (`/api/ai-course-creator/...`).
Auth: JWT or session, plus the existing `user_can_author` check.

| Method and path | Purpose |
|---|---|
| `POST /api/course-from-recording/jobs/` | Create a job with metadata; returns the job and a presigned upload URL. |
| `GET /api/course-from-recording/jobs/?course_id=…` | Latest job for a course, used to resume. |
| `GET /api/course-from-recording/jobs/<id>/` | Status, progress, error and result links (presigned download URLs). |
| `POST /api/course-from-recording/jobs/<id>/start/` | Check the upload exists, then queue the job. |
| `POST /api/course-from-recording/jobs/<id>/retry/` | Queue a failed job again; it continues from the failed step. |
| `POST /api/course-from-recording/internal/jobs/<id>/status/` | **Service only.** Update status, progress, results or error. Requires the shared secret. |
| *later* `GET/PUT …/jobs/<id>/draft/` | Read and save the reviewed outline and text. |
| *later* `POST …/jobs/<id>/publish/` | Write the draft into the course, reusing `ai_course_creator/services/course_builder.py` (`apply_course_json`). |

## Audio service

The Python package `studio_tool_service`, run in its own container. Each Studio tool gets its own
sub-package, so later tools can be added next to it:

```
src/studio_tool_service/
    worker.py                  main loop: take a job from Redis, run it, repeat
    config.py                  settings from environment variables
    storage.py                 S3 download and upload
    status.py                  status updates to the sherab plugin
    course_from_recording/
        job.py                 step order; skips steps whose output is already in S3
        steps/                 clean, speakers, transcribe, correct (later: structure, write, cut)
tutor_plugin/                  adds the container, its limits and settings to docker-compose
Dockerfile
```

The step code is copied from the pipeline repo's scripts and turned into functions that take input and
output paths, without the command-line and `output/` folder logic.

The main loop:

1. Wait for a job id on the Redis list `course_from_recording:queue`. `BLMOVE` moves it to
   `course_from_recording:processing`, so a job is never lost if the container restarts. On start-up,
   anything left in `processing` goes back to the queue.
2. Download the recording from S3 into a temporary folder.
3. Run the steps in order. A step whose output is already in S3 is skipped, so a retried or restarted
   job continues where it stopped.
4. After each step: upload its outputs and `POST` the new status to the plugin.
5. Delete the temporary folder and take the next job.

| Step | Copied from (pipeline repo) | Output in S3 |
|---|---|---|
| Read duration, check the limit | — | `duration_seconds` on the job |
| Clean | `denoise.py` | `clean.wav` |
| Find speakers | `detect_speakers.py` | `speakers/segments.json`, sample clips |
| Transcribe the teacher | `transcribe.py` | `transcript/raw.txt` |
| Correct | `correct_transcript.py` (rules and glossary part) | `transcript/corrected.txt` |
| *later* Structure into sections and units | new, uses an AI model | `draft/curriculum.json` |
| *later* Unit text, key points, questions | new, uses an AI model | `draft/units.json` |
| *later* Cut lesson audio | `cut_unit_audio.py` | `units/<n>-bo.mp3`, `units/<n>-lad.mp3` |

### S3 layout

```
<bucket>/course-from-recording/<job id>/
    original.<ext>
    clean.wav
    speakers/
    transcript/
    draft/
    units/
```

The bucket is private. The browser only gets presigned URLs, which expire.

## Secrets and settings

All values are set in Tutor config on the server, never in git.

| Setting | Used by |
|---|---|
| `MONLAM_API_KEY` | Service only. |
| AWS key and secret, limited to `course-from-recording/*` in the bucket | Service and plugin. |
| `COURSE_FROM_RECORDING_BUCKET` | Service and plugin. |
| `COURSE_FROM_RECORDING_SERVICE_TOKEN` (shared secret for the status API) | Service and plugin. |
| `COURSE_FROM_RECORDING_MAX_MINUTES` (longest recording accepted) | Plugin and service. |
| Redis host and port | Service and plugin; Tutor's existing Redis. |

The Tutor plugin adds the service through the `local-docker-compose-services` patch (and
`local-docker-compose-dev-services` for `tutor dev`), and the plugin settings through
`openedx-cms-common-settings`.

## Cost control

- Monlam costs about **$0.032 per minute** sent (about $2 per hour). Only teacher segments are sent.
- A recording longer than `COURSE_FROM_RECORDING_MAX_MINUTES` is rejected before transcription.
- Minutes sent are stored on each job (`transcribed_minutes`), so monthly spend can be added up.

## Phases

**Phase 1 — recording to transcript.** Proves the whole chain: upload, queue, service, S3, status and resume.

- Plugin: model, API, presigned upload, queue push, status endpoint.
- Service: container, loop, steps: clean, find speakers, transcribe, correct.
- Tutor plugin: service container with limits; settings.
- Tool page: metadata and upload screen, progress screen, transcript screen with downloads.

**Phase 2 — transcript to course draft.**

- Service: structure, write unit text, key points and questions, cut lesson audio.
- Tool page: review screen for the outline, unit text and questions; play lesson clips.

**Phase 3 — draft into the course.**

- Plugin: publish using `apply_course_json`; upload lesson audio as course assets and add the players.

**Later:** listen before and after cleaning, choose which parts to keep, email when a job finishes,
practice plans, Tibetan translations of the tool's labels.

## Open decisions

1. **Telling the voices apart.** `detect_speakers.py` needs example times for each voice.
   - **Option A (recommended for phase 1):** the service splits the speakers into groups by itself
     (`--groups`). The job pauses at `needs_speaker_labels`, the author plays a short sample of each
     group and tags it (teacher, translator, chanting), and the job continues.
   - **Option B:** the author marks example times before processing starts.
2. **AI model** for structuring and writing (phase 2): Gemini (the plugin's existing client) or Claude.
3. **Audio in a lesson** (phase 3): course assets with an HTML audio player, or the
   `xblock-multilingual-text` block.
4. **Two-file recordings:** the current pipeline also handles a teacher and translator recorded
   separately. Phase 1 assumes one recording per job.
