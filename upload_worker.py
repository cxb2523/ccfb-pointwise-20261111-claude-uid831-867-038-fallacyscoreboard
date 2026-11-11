"""Upload worker: media transcription jobs alongside the original text path.

Two job types share one persistent :class:`~job_store.JobStore`:

* ``text``  — the pre-existing flow: the uploaded text goes straight to the
  fallacy scorer. This path is unchanged.
* ``media`` — new flow: ffmpeg probe/normalise, silence segmentation,
  segment-by-segment transcription with checkpoints (see
  :mod:`media_transcribe`), then the timestamped transcript is scored by the
  same fallacy logic, so every detected fallacy keeps its start/end time.

The worker is deliberately dependency-light: ``llm`` and ``whisperx`` are
imported lazily and the scorer/transcriber can be injected, which keeps
``python -m pytest -q`` runnable on machines without torch/ffmpeg/ollama.
"""

import threading
from pathlib import Path

from job_store import (
    JobStore,
    PENDING,
    COMPLETED,
    FAILED,
)
from media_transcribe import (
    MediaTranscriber,
    MediaTranscriptionError,
    SegmentTranscriptionError,
    reap_orphans,
)

MEDIA_JOB = "media"
TEXT_JOB = "text"


def default_scorer(text_path, analysis_path, temp_path):
    """Scorer seam; imports the real LLM pipeline lazily."""
    from llm import detect_fallacies

    detect_fallacies(str(text_path), str(analysis_path), str(temp_path))


class UploadWorker:
    """Polls the job store and runs media/text jobs, surviving restarts."""

    def __init__(
        self,
        store_path,
        work_dir,
        transcriber=None,
        scorer=None,
        max_attempts=3,
        poll_interval=1.0,
        ffmpeg=None,
        ffprobe=None,
    ):
        self.store = JobStore(store_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.poll_interval = poll_interval
        self.scorer = scorer or default_scorer
        self._transcriber = transcriber
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._stop_event = threading.Event()
        self._thread = None
        self._transcriber_instance = None

        # Restart recovery: jobs left RUNNING by a dead process become
        # pending again, and their orphaned wav/segment scratch is removed.
        self.requeued = self.store.requeue_interrupted()
        self.reaped_dirs = reap_orphans(self.work_dir)

    # -- transcriber factory --------------------------------------------------

    def _build_transcriber(self):
        # One model instance per worker process; constructing it lazily keeps
        # text-only deployments and the test-suite free of torch/whisperx.
        if self._transcriber_instance is None:
            self._transcriber_instance = MediaTranscriber(
                self.work_dir,
                transcriber=self._transcriber,
                ffmpeg=self._ffmpeg,
                ffprobe=self._ffprobe,
            )
        return self._transcriber_instance

    # -- enqueue / status (the upload entry points) ---------------------------

    def enqueue_media(self, media_path, job_id=None):
        """Register a media upload (mp4/mov/mkv/mp3/wav) for transcription."""
        media_path = Path(media_path)
        if not media_path.exists():
            raise FileNotFoundError(f"media file not found: {media_path}")
        return self.store.create_job(
            MEDIA_JOB,
            payload={"media_path": str(media_path)},
            job_id=job_id,
        )

    def enqueue_text(self, text, job_id=None):
        """Register a plain-text scoring job (the original behaviour)."""
        if not text or not text.strip():
            raise ValueError("text upload must not be empty")
        return self.store.create_job(
            TEXT_JOB,
            payload={"text": text},
            job_id=job_id,
        )

    def get_status(self, job_id):
        """Poll endpoint: jobs persist in the store across refresh/restart."""
        return self.store.get(job_id)

    # -- scoring --------------------------------------------------------------

    def _run_scorer(self, text_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = output_dir / "fallacy_analysis.json"
        temp_path = output_dir / "score_temp.json"
        self.scorer(text_path, analysis_path, temp_path)
        if not analysis_path.exists():
            raise RuntimeError("scorer produced no fallacy_analysis.json")
        return analysis_path

    # -- job handlers ----------------------------------------------------------

    def _handle_media(self, job):
        media_path = job["payload"]["media_path"]
        transcriber = self._build_transcriber()
        result = transcriber.transcribe(media_path)

        # Same scoring logic as the text path, fed timestamped lines:
        # "{start}-{end} SPEAKER_00: {text}", so each fallacy locates分秒.
        analysis_path = self._run_scorer(
            result["transcript_txt"], Path(result["job_dir"])
        )
        result["fallacy_analysis"] = str(analysis_path)
        return result

    def _handle_text(self, job):
        job_dir = self.work_dir / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        text_path = job_dir / "upload.txt"
        text_path.write_text(job["payload"]["text"], encoding="utf-8")
        analysis_path = self._run_scorer(text_path, job_dir)
        return {
            "text_path": str(text_path),
            "fallacy_analysis": str(analysis_path),
        }

    def process_one(self, job=None):
        """Execute an already-claimed job, or claim one pending job.

        Returns the final job record, or ``None`` when no job was available.
        Exposed for synchronous use (and tests).
        """
        if job is None:
            job = self.store.claim_pending()
        if job is None:
            return None

        try:
            if job["type"] == MEDIA_JOB:
                result = self._handle_media(job)
            elif job["type"] == TEXT_JOB:
                result = self._handle_text(job)
            else:
                raise ValueError(f"unknown job type: {job['type']}")
            self.store.mark_completed(job["id"], result)
        except SegmentTranscriptionError as exc:
            # Checkpoint already holds every completed segment; requeue so the
            # retry only re-runs the failed segment indices.
            if job["attempts"] < self.max_attempts:
                self.store.update(
                    job["id"],
                    status=PENDING,
                    error=(
                        f"{exc}; will resume from checkpoint "
                        f"(attempt {job['attempts']}/{self.max_attempts})"
                    ),
                )
            else:
                self.store.mark_failed(
                    job["id"],
                    f"exhausted {self.max_attempts} attempts; {exc}",
                )
        except MediaTranscriptionError as exc:
            # Deterministic media failures (no audio track, no speech, missing
            # ffmpeg): retrying cannot help, fail fast with a clear message.
            self.store.mark_failed(job["id"], exc)
        except Exception as exc:  # scorer/IO failures: retryable
            if job["attempts"] < self.max_attempts:
                self.store.update(
                    job["id"],
                    status=PENDING,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                self.store.mark_failed(job["id"], exc)
        return self.store.get(job["id"])

    def run_pending(self):
        """Drain every job that was pending when called; returns final records.

        Jobs requeued during the drain (e.g. a segment failure awaiting its
        retry) are NOT picked up in the same pass — call again after the retry
        boundary, as the background loop naturally does on its next poll.
        """
        queued_ids = [
            job["id"]
            for job in self.store.list_jobs()
            if job["status"] == PENDING
        ]
        finished = []
        for job_id in queued_ids:
            job = self.store.claim_job(job_id)
            if job is None:
                continue
            finished.append(self.process_one(job))
        return finished

    # -- background loop -------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_event.is_set():
            job = self.store.claim_pending()
            if job is None:
                self._stop_event.wait(self.poll_interval)
                continue
            self.process_one(job)

    def stop(self, timeout=5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
