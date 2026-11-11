"""Persistent job store for the upload worker.

Jobs live in a single JSON file written atomically (temp file + ``os.replace``)
so polling, page refreshes or a worker restart can never lose a task. Status
transitions are the single source of truth for the worker and any UI polling
the store.
"""

import json
import os
import threading
import uuid
from pathlib import Path

# Lifecycle states.
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

TERMINAL_STATES = {COMPLETED, FAILED}


class JobStore:
    """File-backed job store safe for one worker process plus concurrent reads."""

    def __init__(self, store_path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.store_path.exists():
            self._write_locked({"jobs": {}})

    # -- raw IO ---------------------------------------------------------------

    def _read_locked(self):
        try:
            with open(self.store_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {"jobs": {}}
        if "jobs" not in data or not isinstance(data["jobs"], dict):
            data = {"jobs": {}}
        return data

    def _write_locked(self, data):
        tmp = self.store_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, self.store_path)

    # -- public API -----------------------------------------------------------

    def create_job(self, job_type, payload=None, job_id=None):
        """Register a new job and return its id."""
        with self._lock:
            data = self._read_locked()
            job_id = job_id or uuid.uuid4().hex
            if job_id in data["jobs"]:
                raise ValueError(f"duplicate job id: {job_id}")
            data["jobs"][job_id] = {
                "id": job_id,
                "type": job_type,
                "status": PENDING,
                "payload": payload or {},
                "result": None,
                "error": None,
                "attempts": 0,
            }
            self._write_locked(data)
            return job_id

    def get(self, job_id):
        with self._lock:
            data = self._read_locked()
            job = data["jobs"].get(job_id)
            # Return a copy so callers can never mutate the on-disk state.
            return json.loads(json.dumps(job)) if job else None

    def list_jobs(self):
        with self._lock:
            data = self._read_locked()
            return [json.loads(json.dumps(job)) for job in data["jobs"].values()]

    def update(self, job_id, **fields):
        """Patch arbitrary fields of a job and persist atomically."""
        with self._lock:
            data = self._read_locked()
            job = data["jobs"].get(job_id)
            if job is None:
                raise KeyError(f"unknown job id: {job_id}")
            job.update(fields)
            self._write_locked(data)
            return json.loads(json.dumps(job))

    def mark_running(self, job_id):
        with self._lock:
            data = self._read_locked()
            job = data["jobs"].get(job_id)
            if job is None:
                raise KeyError(f"unknown job id: {job_id}")
            job["status"] = RUNNING
            job["attempts"] = int(job.get("attempts", 0)) + 1
            self._write_locked(data)
            return json.loads(json.dumps(job))

    def mark_completed(self, job_id, result):
        return self.update(job_id, status=COMPLETED, result=result, error=None)

    def mark_failed(self, job_id, error):
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        return self.update(job_id, status=FAILED, error=message)

    def claim_job(self, job_id):
        """Claim one specific pending job by id (``None`` if no longer pending)."""
        with self._lock:
            data = self._read_locked()
            job = data["jobs"].get(job_id)
            if job is None or job["status"] != PENDING:
                return None
            job["status"] = RUNNING
            job["attempts"] = int(job.get("attempts", 0)) + 1
            self._write_locked(data)
            return json.loads(json.dumps(job))

    def claim_pending(self, job_type=None):
        """Atomically claim one pending job (used by worker loops)."""
        with self._lock:
            data = self._read_locked()
            for job in data["jobs"].values():
                if job["status"] != PENDING:
                    continue
                if job_type is not None and job["type"] != job_type:
                    continue
                job["status"] = RUNNING
                job["attempts"] = int(job.get("attempts", 0)) + 1
                self._write_locked(data)
                return json.loads(json.dumps(job))
            return None

    def requeue_interrupted(self):
        """Move jobs left RUNNING by a crashed/restarted worker back to pending.

        Terminal jobs are untouched; pending/completed/failed survive a restart
        because the whole store is file-backed.
        """
        with self._lock:
            data = self._read_locked()
            requeued = []
            for job in data["jobs"].values():
                if job["status"] == RUNNING:
                    job["status"] = PENDING
                    requeued.append(job["id"])
            if requeued:
                self._write_locked(data)
            return requeued
