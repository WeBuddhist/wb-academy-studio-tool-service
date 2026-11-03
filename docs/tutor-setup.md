# Running the service with Tutor

The service runs as one more container next to Open edX, in `tutor local` and `tutor dev`. Tutor
learns about it from the `studio_tool_service.py` plugin, which lives in the Tutor plugins folder
(`tutor plugins printroot`) next to the other WeBuddhist plugins, not in this repo.

| What the plugin adds | Where it ends up |
|---|---|
| The `studio-tool-service` container, limited to 4 GB of memory and 2 CPUs | `env/local/docker-compose.yml` |
| In `tutor dev`: this repo's `src/` mounted into the container | `env/dev/docker-compose.yml` |
| `STUDIO_TOOL_SERVICE` settings for the sherab plugin (queue, bucket, shared secret) | Studio settings |
| `tutor images build studio-tool-service` | builds the image from this repo's `Dockerfile` |

## Set up

1. Put `studio_tool_service.py` in the Tutor plugins folder, get this repo onto the machine, and turn
   the plugin on:

   ```bash
   git clone https://github.com/Webuddhist-tech/wb-academy-studio-tool-service.git
   tutor plugins enable studio_tool_service
   ```

2. Tell Tutor where the repo is, and set the Monlam key:

   ```bash
   tutor config save \
     --set STUDIO_TOOL_SERVICE_SOURCE_DIR="$PWD/wb-academy-studio-tool-service" \
     --set STUDIO_TOOL_SERVICE_MONLAM_API_KEY=...
   ```

   In production, also set a separate AWS key that can only reach the `studio-tools/` folder:
   `--set STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID=... --set STUDIO_TOOL_SERVICE_AWS_SECRET_ACCESS_KEY=...`

3. Build the image (about 2–3 GB, mostly PyTorch) and start:

   ```bash
   tutor images build studio-tool-service
   tutor local start -d          # or: tutor dev start -d
   ```

   The build may print `ERROR: failed to configure registry cache importer`. That is harmless: there
   is no image registry with a cached copy, so Docker builds everything locally.

4. Check it is waiting for jobs:

   ```bash
   tutor local logs -f studio-tool-service
   ```

After a `git pull`: rebuild the image, then `tutor local start -d` again. In `tutor dev`, code changes
only need `tutor dev restart studio-tool-service`; rebuild only when `requirements.txt` or the
`Dockerfile` change.

## Settings

| Setting | Default |
|---|---|
| `STUDIO_TOOL_SERVICE_SOURCE_DIR` | empty; needed to build the image |
| `STUDIO_TOOL_SERVICE_BUCKET` | `S3_STORAGE_BUCKET` (the private Open edX bucket) |
| `STUDIO_TOOL_SERVICE_PREFIX` | `studio-tools` |
| `STUDIO_TOOL_SERVICE_AWS_REGION` | `S3_REGION` |
| `STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID` / `..._SECRET_ACCESS_KEY` | the Open edX key |
| `STUDIO_TOOL_SERVICE_MONLAM_API_KEY` | empty |
| `STUDIO_TOOL_SERVICE_REDIS_DB` | `3` (Open edX uses 0 and 1) |
| `STUDIO_TOOL_SERVICE_QUEUE` | `studio_tool_service:jobs` |
| `STUDIO_TOOL_SERVICE_CMS_URL` | `http://cms:8000` |
| `STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES` | `240` |
| `STUDIO_TOOL_SERVICE_MEMORY_LIMIT` / `..._CPUS` | `4g` / `2` |
| `STUDIO_TOOL_SERVICE_TOKEN` | generated once; the shared secret for status updates |

The service code reads these as environment variables (see `src/studio_tool_service/config.py`), so
a setting renamed in the plugin must be renamed there too.
