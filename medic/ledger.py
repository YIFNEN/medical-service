from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import RawArtifact


_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.RLock] = {}


@contextmanager
def exclusive_file_lock(lock_path: Path):
    resolved = str(lock_path.resolve())
    with _LOCK_REGISTRY_GUARD:
        thread_lock = _LOCK_REGISTRY.setdefault(resolved, threading.RLock())
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock_file.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def deterministic_id(prefix: str, value: Any) -> str:
    digest = sha256_digest(canonical_json_bytes(value)).removeprefix("sha256:")
    return f"{prefix}-{digest}"


def _is_strict_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_strict_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_strict_json(item)
            for key, item in value.items()
        )
    return False


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


class ContentAddressedStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.raw_root = self.root / "raw"
        self.metadata_root = self.root / "metadata"
        self.compiled_root = self.root / "compiled"

    def register_raw(self, case_id: str, source_path: Path) -> RawArtifact:
        source_path = Path(source_path)
        content = source_path.read_bytes()
        content_hash = sha256_digest(content)
        hash_value = content_hash.removeprefix("sha256:")
        artifact_id = f"artifact-{hash_value}"
        raw_path = self.raw_root / hash_value
        metadata_path = self.metadata_root / f"{artifact_id}.json"

        with exclusive_file_lock(self.root / "locks" / f"artifact-{hash_value}.lock"):
            return self._register_locked(
                case_id, source_path, content, content_hash, artifact_id, raw_path, metadata_path
            )

    def _register_locked(
        self,
        case_id: str,
        source_path: Path,
        content: bytes,
        content_hash: str,
        artifact_id: str,
        raw_path: Path,
        metadata_path: Path,
    ) -> RawArtifact:

        if raw_path.exists():
            if sha256_digest(raw_path.read_bytes()) != content_hash:
                raise RuntimeError(f"content-address collision for {artifact_id}")
        else:
            _atomic_write(raw_path, content)

        if metadata_path.exists():
            artifact = RawArtifact.model_validate_json(metadata_path.read_text(encoding="utf-8"))
            if artifact.artifact_id != artifact_id or artifact.content_hash != content_hash:
                raise RuntimeError(f"artifact metadata integrity mismatch for {artifact_id}")
            if artifact.case_id != case_id:
                raise ValueError("the same raw artifact cannot be registered to a different case")
            return artifact

        media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
        artifact = RawArtifact(
            artifact_id=artifact_id,
            case_id=case_id,
            content_hash=content_hash,
            media_type=media_type,
            source_name=source_path.name,
            ingested_at=datetime.now(UTC),
        )
        _atomic_write(
            metadata_path,
            canonical_json_bytes(artifact.model_dump(mode="json")),
        )
        return artifact

    def key_lock(self, namespace: str, key: str):
        safe_key = sha256_digest(key.encode("utf-8")).removeprefix("sha256:")
        return exclusive_file_lock(self.root / "locks" / f"{namespace}-{safe_key}.lock")

    def compiled_path(self, compile_id: str) -> Path:
        return self.compiled_root / f"{compile_id}.json"

    def read_raw(self, artifact: RawArtifact) -> Path:
        hash_value = artifact.content_hash.removeprefix("sha256:")
        path = self.raw_root / hash_value
        if not path.exists() or sha256_digest(path.read_bytes()) != artifact.content_hash:
            raise RuntimeError(f"immutable raw artifact is missing or corrupt: {artifact.artifact_id}")
        return path

    def save_compiled(self, compile_id: str, payload: dict[str, Any]) -> None:
        target = self.compiled_path(compile_id)
        manifest_bytes = canonical_json_bytes(payload)
        encoded = canonical_json_bytes(
            {"manifest": payload, "manifest_hash": sha256_digest(manifest_bytes)}
        )
        if target.exists():
            if target.read_bytes() != encoded:
                raise RuntimeError(f"immutable compile manifest conflict for {compile_id}")
            return
        _atomic_write(target, encoded)


class AppendOnlyJsonlLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _records_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            records = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("ledger corruption requires explicit recovery") from exc
        seen: dict[str, dict[str, Any]] = {}
        natural_keys: set[tuple[str, str]] = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "record_id", "kind", "natural_key", "payload"
            }:
                raise RuntimeError("ledger corruption: invalid record schema")
            if not all(
                isinstance(record[field], str) and record[field]
                for field in ("record_id", "kind", "natural_key")
            ) or not isinstance(record["payload"], dict):
                raise RuntimeError("ledger corruption: invalid record field type")
            record_id = record["record_id"]
            expected_id = deterministic_id(
                "record",
                {"kind": record["kind"], "natural_key": record["natural_key"]},
            )
            if record_id != expected_id:
                raise RuntimeError(f"ledger corruption: forged record id {record_id}")
            if record_id in seen:
                raise RuntimeError(f"duplicate or conflicting ledger record: {record_id}")
            natural_key = (record["kind"], record["natural_key"])
            if natural_key in natural_keys:
                raise RuntimeError(f"duplicate or conflicting ledger natural key: {natural_key}")
            seen[record_id] = record
            natural_keys.add(natural_key)
        return records

    def records(self) -> list[dict[str, Any]]:
        with self._exclusive_lock():
            return self._records_unlocked()

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        with exclusive_file_lock(lock_path):
            yield

    def append_once(self, kind: str, natural_key: str, payload: dict[str, Any]) -> str:
        return self.append_batch_once(((kind, natural_key, payload),))[0]

    def append_batch_once(
        self,
        entries: tuple[tuple[str, str, dict[str, Any]], ...],
    ) -> tuple[str, ...]:
        prepared: list[dict[str, Any]] = []
        for kind, natural_key, payload in entries:
            prepared.append(self._prepare_record(kind, natural_key, payload))
        record_ids = [record["record_id"] for record in prepared]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("ledger batch contains duplicate natural keys")
        with self._exclusive_lock():
            existing_by_id = {
                item["record_id"]: item for item in self._records_unlocked()
            }
            to_append: list[dict[str, Any]] = []
            for record in prepared:
                existing = existing_by_id.get(record["record_id"])
                if existing is not None:
                    if canonical_json_bytes(existing) != canonical_json_bytes(record):
                        raise RuntimeError(
                            f"append-only ledger conflict for {record['record_id']}"
                        )
                    continue
                to_append.append(record)
            if to_append:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                encoded = b"".join(canonical_json_bytes(record) + b"\n" for record in to_append)
                with self.path.open("ab") as ledger_file:
                    ledger_file.write(encoded)
                    ledger_file.flush()
                    os.fsync(ledger_file.fileno())
        return tuple(record_ids)

    @staticmethod
    def _prepare_record(kind: str, natural_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind or not isinstance(natural_key, str) or not natural_key:
            raise ValueError("ledger record kind and natural_key must be non-empty strings")
        if not isinstance(payload, dict):
            raise ValueError("ledger record payload must be a JSON object")
        if not _is_strict_json(payload):
            raise ValueError("ledger record payload must use strict JSON value types")
        try:
            canonical_json_bytes(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("ledger record payload must contain finite JSON values") from exc
        record_id = deterministic_id(
            "record",
            {"kind": kind, "natural_key": natural_key},
        )
        record = {
            "record_id": record_id,
            "kind": kind,
            "natural_key": natural_key,
            "payload": payload,
        }
        return record

    def key_lock(self, namespace: str, key: str):
        safe_key = sha256_digest(key.encode("utf-8")).removeprefix("sha256:")
        return exclusive_file_lock(
            self.path.with_name(f"{self.path.name}.{namespace}-{safe_key}.lock")
        )
