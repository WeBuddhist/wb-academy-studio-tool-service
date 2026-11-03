"""
Tutor plugin for the studio tool service: the container that runs the heavy background jobs of
WeBuddhist Studio tools (the first one is "course from a recording").

It adds the container to `tutor local` / `tutor dev`, lets `tutor images build studio-tool-service`
build the image from a checkout of this repo, and gives Studio (the sherab plugin) the queue, bucket
and shared-secret settings. Setup steps: tutor_plugin/README.md.
"""
from tutor import hooks

VERSION = "0.0.1"

hooks.Filters.CONFIG_DEFAULTS.add_items([
    ("STUDIO_TOOL_SERVICE_VERSION", VERSION),
    ("STUDIO_TOOL_SERVICE_DOCKER_IMAGE", "studio-tool-service:{{ STUDIO_TOOL_SERVICE_VERSION }}"),
    # Absolute path of this repo's checkout on the machine; the image is built from it.
    ("STUDIO_TOOL_SERVICE_SOURCE_DIR", ""),
    # Files go in a folder of the private Open edX bucket.
    ("STUDIO_TOOL_SERVICE_BUCKET", "{{ S3_STORAGE_BUCKET|default('') }}"),
    ("STUDIO_TOOL_SERVICE_PREFIX", "studio-tools"),
    ("STUDIO_TOOL_SERVICE_AWS_REGION", "{{ S3_REGION|default('') }}"),
    # Defaults to the Open edX key. In production, set a key that only reaches the studio-tools folder.
    ("STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID", "{{ OPENEDX_AWS_ACCESS_KEY }}"),
    ("STUDIO_TOOL_SERVICE_AWS_SECRET_ACCESS_KEY", "{{ OPENEDX_AWS_SECRET_ACCESS_KEY }}"),
    ("STUDIO_TOOL_SERVICE_MONLAM_API_KEY", ""),
    # Tutor's Redis; Open edX uses databases 0 (Celery) and 1 (cache).
    ("STUDIO_TOOL_SERVICE_REDIS_DB", 3),
    (
        "STUDIO_TOOL_SERVICE_REDIS_URL",
        "redis://{% if REDIS_USERNAME and REDIS_PASSWORD %}{{ REDIS_USERNAME }}:{{ REDIS_PASSWORD }}@{% endif %}"
        "{{ REDIS_HOST }}:{{ REDIS_PORT }}/{{ STUDIO_TOOL_SERVICE_REDIS_DB }}",
    ),
    ("STUDIO_TOOL_SERVICE_QUEUE", "studio_tool_service:jobs"),
    ("STUDIO_TOOL_SERVICE_CMS_URL", "http://cms:8000"),
    ("STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES", 240),
    # The server is shared with Open edX: keep the service inside these limits.
    ("STUDIO_TOOL_SERVICE_MEMORY_LIMIT", "4g"),
    ("STUDIO_TOOL_SERVICE_CPUS", 2),
])

# Generated once by `tutor config save` and kept in config.yml.
hooks.Filters.CONFIG_UNIQUE.add_items([
    ("STUDIO_TOOL_SERVICE_TOKEN", "{{ 32|random_string }}"),
])

hooks.Filters.ENV_PATCHES.add_items([
    (
        "local-docker-compose-services",
        """
studio-tool-service:
  image: {{ STUDIO_TOOL_SERVICE_DOCKER_IMAGE }}
  environment:
    REDIS_URL: "{{ STUDIO_TOOL_SERVICE_REDIS_URL }}"
    QUEUE_NAME: "{{ STUDIO_TOOL_SERVICE_QUEUE }}"
    S3_BUCKET: "{{ STUDIO_TOOL_SERVICE_BUCKET }}"
    S3_PREFIX: "{{ STUDIO_TOOL_SERVICE_PREFIX }}"
    AWS_DEFAULT_REGION: "{{ STUDIO_TOOL_SERVICE_AWS_REGION }}"
    AWS_ACCESS_KEY_ID: "{{ STUDIO_TOOL_SERVICE_AWS_ACCESS_KEY_ID }}"
    AWS_SECRET_ACCESS_KEY: "{{ STUDIO_TOOL_SERVICE_AWS_SECRET_ACCESS_KEY }}"
    MONLAM_API_KEY: "{{ STUDIO_TOOL_SERVICE_MONLAM_API_KEY }}"
    CMS_URL: "{{ STUDIO_TOOL_SERVICE_CMS_URL }}"
    SERVICE_TOKEN: "{{ STUDIO_TOOL_SERVICE_TOKEN }}"
    MAX_RECORDING_MINUTES: "{{ STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES }}"
  mem_limit: {{ STUDIO_TOOL_SERVICE_MEMORY_LIMIT }}
  cpus: {{ STUDIO_TOOL_SERVICE_CPUS }}
  restart: unless-stopped
  {% if RUN_REDIS %}depends_on:
    - redis{% endif %}
""",
    ),
    (
        # In `tutor dev`, run the code from the checkout, so edits only need a container restart.
        "local-docker-compose-dev-services",
        """
{% if STUDIO_TOOL_SERVICE_SOURCE_DIR %}
studio-tool-service:
  volumes:
    - "{{ STUDIO_TOOL_SERVICE_SOURCE_DIR }}/src:/app/src:ro"
{% endif %}
""",
    ),
    (
        # Read by the sherab plugin's course_from_recording app.
        "openedx-cms-common-settings",
        """
STUDIO_TOOL_SERVICE = {
    "REDIS_URL": "{{ STUDIO_TOOL_SERVICE_REDIS_URL }}",
    "QUEUE": "{{ STUDIO_TOOL_SERVICE_QUEUE }}",
    "BUCKET": "{{ STUDIO_TOOL_SERVICE_BUCKET }}",
    "PREFIX": "{{ STUDIO_TOOL_SERVICE_PREFIX }}",
    "TOKEN": "{{ STUDIO_TOOL_SERVICE_TOKEN }}",
    "MAX_RECORDING_MINUTES": {{ STUDIO_TOOL_SERVICE_MAX_RECORDING_MINUTES }},
}
""",
    ),
])


@hooks.Filters.IMAGES_BUILD.add()
def _build_from_source_dir(images, config):
    source_dir = config["STUDIO_TOOL_SERVICE_SOURCE_DIR"]
    if source_dir:
        # Tutor joins this onto its env folder; an absolute path stays as it is.
        images.append(("studio-tool-service", source_dir, "{{ STUDIO_TOOL_SERVICE_DOCKER_IMAGE }}", ()))
    return images
