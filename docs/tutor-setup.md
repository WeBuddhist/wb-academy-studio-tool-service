# Running the service with Tutor

The service runs as one more container next to Open edX, in `tutor local` and `tutor dev`. Tutor
learns about it from the `studio_tool_service.py` plugin, which lives in the Tutor plugins folder
(`tutor plugins printroot`) next to the other WeBuddhist plugins, not in this repo.

| What the plugin adds | Where it ends up |
|---|---|
| The `studio-tool-service` container, limited to 4 GB of memory and 2 CPUs | `env/local/docker-compose.yml` |
| In `tutor dev`: this repo's `src/` mounted into the container | `env/dev/docker-compose.yml` |
| `STUDIO_TOOL_SERVICE` settings for the sherab plugin (queue, bucket, shared secret) | Studio settings |
| `tutor images build studio-tool-service` | builds the image from this repo's `Dockerfile`: from a checkout on the machine, or straight from GitHub |

There are two ways to set it up:

- **On a laptop** (below): the image is built from a checkout of this repo, and in `tutor dev` the
  container runs that checkout's code, so edits only need a restart.
- **On a server** ([Deploying to a server](#deploying-to-a-server)): no checkout. Tutor builds the image
  straight from GitHub at a branch, the way it builds edx-platform.

## Set up on a laptop

1. Put `studio_tool_service.py` in the Tutor plugins folder, get this repo onto the machine, and turn
   the plugin on:

   ```bash
   git clone https://github.com/Webuddhist-tech/wb-academy-studio-tool-service.git
   tutor plugins enable studio_tool_service
   ```

2. Tell Tutor where the repo is, and set the Monlam key (for Tibetan transcription). Writing the course
   uses the OpenRouter key Tutor already has for the AI course creator (`OPENROUTER_API_KEY`):

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

## Deploying to a server

On a server there is no checkout and no local path: `config.yml` names the repository and the branch,
next to the `OPENEDX_EXTRA_PIP_REQUIREMENTS` of the other WeBuddhist code.

1. Put `studio_tool_service.py` in the server's Tutor plugins folder (`tutor plugins printroot`) and
   turn it on with `tutor plugins enable studio_tool_service`.

2. Add to `config.yml` (then run `tutor config save`). Use the branch for that server:

   ```yaml
   STUDIO_TOOL_SERVICE_REPOSITORY_VERSION: wba-ulmo1-stage   # wba-ulmo1-prod on production
   STUDIO_TOOL_SERVICE_MONLAM_API_KEY: ...
   STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID: ...                # a key that only reaches studio-tools/
   STUDIO_TOOL_SERVICE_AWS_SECRET_ACCESS_KEY: ...
   ```

   Leave `STUDIO_TOOL_SERVICE_SOURCE_DIR` out: when it is set, it wins over the repository.
   `STUDIO_TOOL_SERVICE_REPOSITORY` already points at this repo on GitHub; set it only to build from a
   fork.

3. Build and start:

   ```bash
   tutor images build studio-tool-service
   tutor local start -d
   tutor local logs -f studio-tool-service    # should say it is waiting for jobs
   ```

   Docker downloads the branch from GitHub itself; nothing is cloned onto the server.

**Releasing a change:** merge it into the server's branch, then on the server run
`tutor images build studio-tool-service` and `tutor local start -d`. The build fetches the branch again,
so it always gets the latest commit. Before stopping the old container, check in Studio that no
recording is being processed; a job that is cut off is picked up again when the service starts, but the
step it was on starts over.

**If the repository becomes private:** Docker then needs a GitHub token to download it. Give it a
read-only token through the `GIT_AUTH_TOKEN` build secret:

```bash
GIT_AUTH_TOKEN=... tutor images build studio-tool-service \
  --docker-arg=--secret --docker-arg=id=GIT_AUTH_TOKEN,env=GIT_AUTH_TOKEN
```

**Later, with automatic deploys:** a GitHub Action can build the image when a branch changes and push
it to a registry (such as `ghcr.io`). The server then only needs
`STUDIO_TOOL_SERVICE_DOCKER_IMAGE: ghcr.io/...:wba-ulmo1-stage` and `tutor images pull`, and no
building happens on the server.

## Settings

| Setting | Default |
|---|---|
| `STUDIO_TOOL_SERVICE_SOURCE_DIR` | empty; on a laptop, the checkout to build from (and to mount in `tutor dev`) |
| `STUDIO_TOOL_SERVICE_REPOSITORY` | this repo on GitHub; built from when `SOURCE_DIR` is empty |
| `STUDIO_TOOL_SERVICE_REPOSITORY_VERSION` | `main`; the branch or tag to build (`wba-ulmo1-stage` / `wba-ulmo1-prod` on the servers) |
| `STUDIO_TOOL_SERVICE_DOCKER_IMAGE` | `studio-tool-service:0.0.1`; the image the container runs |
| `STUDIO_TOOL_SERVICE_BUCKET` | `S3_STORAGE_BUCKET` (the private Open edX bucket) |
| `STUDIO_TOOL_SERVICE_PREFIX` | `studio-tools` |
| `STUDIO_TOOL_SERVICE_PUBLIC_BUCKET` | `S3_PARTNER_LOGO_BUCKET` (`sherab-public-assets-dev` / `sherab-public-assets`); where Studio copies the finished course's lesson audio |
| `STUDIO_TOOL_SERVICE_PUBLIC_DOMAIN` | `S3_PARTNER_LOGO_CUSTOM_DOMAIN`; the address the course's players load that audio from |
| `STUDIO_TOOL_SERVICE_AWS_REGION` | `S3_REGION` |
| `STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID` / `..._SECRET_ACCESS_KEY` | the Open edX key |
| `STUDIO_TOOL_SERVICE_MONLAM_API_KEY` | empty; needed for Tibetan |
| `STUDIO_TOOL_SERVICE_OPENROUTER_API_KEY` / `..._OPENROUTER_MODEL` | Tutor's `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` (the AI course creator's; model `google/gemini-2.5-flash-lite` if unset); needed to suggest a course |
| `STUDIO_TOOL_SERVICE_REDIS_DB` | `3` (Open edX uses 0 and 1) |
| `STUDIO_TOOL_SERVICE_QUEUE` | `studio_tool_service:jobs` |
| `STUDIO_TOOL_SERVICE_CMS_URL` | `http://cms:8000` |
| `STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES` | `240` |
| `STUDIO_TOOL_SERVICE_MEMORY_LIMIT` / `..._CPUS` | `4g` / `2` |
| `STUDIO_TOOL_SERVICE_TOKEN` | generated once; the shared secret for status updates |

The service code reads these as environment variables (see `src/studio_tool_service/config.py`), so
a setting renamed in the plugin must be renamed there too.
