"""The S3 bucket every tool keeps its files in. AWS key and region come from the standard AWS_* variables."""

import json
from pathlib import Path
from typing import Any, Callable, Optional

import boto3
from botocore.exceptions import ClientError


class Storage:
    def __init__(self, bucket: str, client=None):
        self.bucket = bucket
        self.client = client or boto3.client("s3")

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def download(self, key: str, path: Path, on_progress: Callable[[float], None] = lambda done: None) -> Path:
        """Download ``key`` to ``path``, calling ``on_progress`` with the share done (0-1) as it goes."""
        total = self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"] or 1
        received = [0]

        def on_bytes(count: int) -> None:
            received[0] += count
            on_progress(min(1.0, received[0] / total))

        self.client.download_file(self.bucket, key, str(path), Callback=on_bytes)
        return path

    def upload(self, path: Path, key: str, content_type: str = "application/octet-stream") -> None:
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})

    def write_text(self, key: str, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

    def write_json(self, key: str, data: Any) -> None:
        self.write_text(key, json.dumps(data, ensure_ascii=False, indent=2), "application/json")

    def delete_folder(self, folder: str, keep: str = "") -> int:
        """Delete every file under ``folder``/, except those under ``folder``/``keep``. Returns how many."""
        prefix = f"{folder}/"
        keys = [
            item["Key"]
            for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
            for item in page.get("Contents", [])
            if not (keep and item["Key"].startswith(prefix + keep))
        ]
        for start in range(0, len(keys), 1000):  # S3 deletes at most 1000 files per request
            batch = [{"Key": key} for key in keys[start : start + 1000]]
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})
        return len(keys)

    def read_json(self, key: str) -> Optional[Any]:
        """The JSON file at ``key``, or None if there is none."""
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return json.loads(body)
