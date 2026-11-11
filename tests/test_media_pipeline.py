"""Tests for the media transcription background task.

The three cases required by the upload-worker change are all here:

* ``test_no_audio_track_video_fails_explicitly`` — video without an audio stream
  must raise a clear NoAudioTrackError instead of scoring empty text into 0.
* ``test_short_audio_is_transcribed_and_scored_with_timestamps`` — the short
  audio happy path, end to end through the worker and the (fake) scorer.
* ``test_checkpoint_resume_only_reruns_failed_segments`` — a flaky transcriber
  fails one segment; the retry resumes from the hash+index checkpoint.

ffmpeg/ffprobe and the speech model are faked through the module seams, so the
suite needs neither binaries nor torch.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import media_transcribe as mt
from job_store import JobStore, PENDING, COMPLETED, FAILED
from media_transcribe import (
    MediaTranscriber,
    NoAudioTrackError,
    FfmpegNotFoundError,
    SegmentTranscriptionError,
    reap_orphans,
)
from upload_worker import UploadWorker, MEDIA_JOB, TEXT_JOB


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeFfmpeg:
    """Stand-in for ffmpeg/ffprobe driven through the module-level seams."""

    def __init__(self, has_audio=True, duration=10.0, silences=()):
        self.has_audio = has_audio
        self.duration = duration
        self.silences = list(silences)
        self.normalize_calls = 0
        self.cut_calls = []

    def install(self, monkeypatch, module=mt):
        monkeypatch.setattr(module, "_which", self._which)
        monkeypatch.setattr(module, "_run_command", self._run_command)
        monkeypatch.setattr(module.subprocess, "run", self._subprocess_run)

    @staticmethod
    def _which(name):
        return f"C:/fakebin/{name}.exe"

    def _run_command(self, cmd):
        cmd = [str(part) for part in cmd]
        binary = Path(cmd[0]).name
        if "ffprobe" in binary:
            streams = [{"codec_type": "video", "codec_name": "h264"}]
            if self.has_audio:
                streams.append({"codec_type": "audio", "codec_name": "aac"})
            payload = {"format": {"duration": str(self.duration)}, "streams": streams}
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

        # normalize / cut: materialise the declared output file.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE-PCM")
        if "-t" in cmd:  # segment cut uses -ss/-t
            self.cut_calls.append(cmd)
        else:
            self.normalize_calls += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _subprocess_run(self, cmd, **kwargs):
        cmd = [str(part) for part in cmd]
        if any("silencedetect" in part for part in cmd):
            lines = []
            for start, end in self.silences:
                lines.append(f"[silencedetect @ 0xabc] silence_start: {start}")
                lines.append(
                    f"[silencedetect @ 0xabc] silence_end: {end} "
                    f"| silence_duration: {end - start:.3f}"
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="\n".join(lines))
        raise AssertionError(f"unexpected subprocess.run invocation: {cmd}")


class FlakyTranscriber:
    """Fake segment model; configured indices fail on their first attempt."""

    def __init__(self, fail_once_indices=()):
        self.fail_remaining = set(fail_once_indices)
        self.calls = []

    def __call__(self, segment_wav):
        index = int(Path(segment_wav).stem.split("_")[1])
        self.calls.append(index)
        if index in self.fail_remaining:
            self.fail_remaining.discard(index)
            raise RuntimeError("simulated transcription backend crash")
        return {
            "text": f"hello segment {index}",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": f"hello segment {index}"}
            ],
        }


def fake_scorer(text_path, analysis_path, temp_path):
    """Mimic llm.detect_fallacies file IO while preserving timestamps."""
    lines = Path(text_path).read_text(encoding="utf-8").strip().splitlines()
    findings = []
    for line in lines:
        span, rest = line.split(" ", 1)
        start, end = span.split("-")
        findings.append(
            {
                "text_segment": rest.split(":", 1)[1].strip(),
                "fallacy_explanation": "unit-test finding",
                "fallacy_type": ["ad hominem"],
                "speaker": rest.split(":", 1)[0],
                "start": float(start),
                "end": float(end),
                "gif_query": "",
            }
        )
    Path(analysis_path).write_text(json.dumps(findings, indent=2), encoding="utf-8")


def make_worker(tmp_path, transcriber, ffmpeg_fake, scorer=fake_scorer):
    store_path = tmp_path / "jobs" / "job_store.json"
    work_dir = tmp_path / "media_work"
    return UploadWorker(
        store_path=store_path,
        work_dir=work_dir,
        transcriber=transcriber,
        scorer=scorer,
        ffmpeg="C:/fakebin/ffmpeg.exe",
        ffprobe="C:/fakebin/ffprobe.exe",
    )


# ---------------------------------------------------------------------------
# Case 1: video without an audio track
# ---------------------------------------------------------------------------


def test_no_audio_track_video_fails_explicitly(tmp_path, monkeypatch):
    fake = FakeFfmpeg(has_audio=False, duration=8.0)
    fake.install(monkeypatch)

    video = tmp_path / "silent.mp4"
    video.write_bytes(b"fake-video-container")

    worker = make_worker(tmp_path, FlakyTranscriber(), fake)
    job_id = worker.enqueue_media(video)
    jobs = worker.run_pending()

    assert len(jobs) == 1
    job = worker.get_status(job_id)
    assert job["status"] == FAILED
    assert "no audio track" in job["error"].lower()

    # The failed job must still be visible after a "page refresh": a brand new
    # store instance reading the same file sees the same terminal record.
    refreshed = JobStore(worker.store.store_path).get(job_id)
    assert refreshed["status"] == FAILED

    # No scratch audio may linger, and no empty transcript may be scored.
    job_dirs = list(Path(worker.work_dir).glob("*"))
    for job_dir in job_dirs:
        assert not (job_dir / "normalized.wav").exists()
        assert not (job_dir / "segments").exists()
        assert not (job_dir / "worker.lock").exists()
        assert not (job_dir / "transcript.txt").exists()


def test_long_silence_is_rejected_not_scored_as_zero(tmp_path, monkeypatch):
    # Audio exists but one silence spans the whole file -> no speech segments.
    fake = FakeFfmpeg(has_audio=True, duration=10.0, silences=[(0.0, 10.0)])
    fake.install(monkeypatch)

    audio = tmp_path / "music_only.wav"
    audio.write_bytes(b"fake-wav")

    worker = make_worker(tmp_path, FlakyTranscriber(), fake)
    job_id = worker.enqueue_media(audio)
    worker.run_pending()

    job = worker.get_status(job_id)
    assert job["status"] == FAILED
    assert "speech" in job["error"].lower() or "silence" in job["error"].lower()


def test_missing_ffmpeg_gets_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "_which", lambda name: None)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    with pytest.raises(FfmpegNotFoundError) as exc_info:
        MediaTranscriber(tmp_path / "work").transcribe(media)
    assert "ffmpeg" in str(exc_info.value).lower()
    assert "install" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Case 2: short audio is transcribed and scored with mapped timestamps
# ---------------------------------------------------------------------------


def test_short_audio_is_transcribed_and_scored_with_timestamps(tmp_path, monkeypatch):
    fake = FakeFfmpeg(has_audio=True, duration=3.0, silences=[])
    fake.install(monkeypatch)

    audio = tmp_path / "interview.mp3"
    audio.write_bytes(b"fake-mp3")

    transcriber = FlakyTranscriber()
    worker = make_worker(tmp_path, transcriber, fake)
    job_id = worker.enqueue_media(audio)
    worker.run_pending()

    job = worker.get_status(job_id)
    assert job["status"] == COMPLETED, job["error"]
    assert job["type"] == MEDIA_JOB

    result = job["result"]
    transcript_lines = Path(result["transcript_txt"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(transcript_lines) == 1
    # llm.detect_fallacies line contract: "{start}-{end} SPEAKER_00: {text}"
    start, end, speaker, text = mt.re.match(
        r"(\d+\.\d+)-(\d+\.\d+) (\S+): (.+)", transcript_lines[0]
    ).groups()
    assert speaker == "SPEAKER_00"
    assert text == "hello segment 0"
    assert float(start) == 0.0 and float(end) == 1.0

    # Every fallacy row keeps its original-timeline position (分秒 locatable).
    findings = json.loads(Path(result["fallacy_analysis"]).read_text(encoding="utf-8"))
    assert findings[0]["start"] == 0.0
    assert findings[0]["end"] == 1.0

    transcript_json = json.loads(Path(result["transcript_json"]).read_text(encoding="utf-8"))
    seg = transcript_json["segments"][0]
    assert seg["start_clock"] == "00:00.0" and seg["end_clock"] == "00:01.0"
    assert transcript_json["source_hash"]

    # Scratch files cleaned on the normal completion path.
    job_dir = Path(result["job_dir"])
    assert not (job_dir / "normalized.wav").exists()
    assert not (job_dir / "segments").exists()
    assert not (job_dir / "worker.lock").exists()
    # Checkpoint and transcript outputs are retained for audit/resume.
    assert (job_dir / "checkpoint.json").exists()

    # Refresh/poll persistence: a fresh worker on the same paths sees the job.
    polling = JobStore(worker.store.store_path).get(job_id)
    assert polling["status"] == COMPLETED


# ---------------------------------------------------------------------------
# Case 3: checkpoint resume — retries only re-run failed segments
# ---------------------------------------------------------------------------


def test_checkpoint_resume_only_reruns_failed_segments(tmp_path, monkeypatch):
    # 10s audio with a silence in the middle -> two speech segments:
    # [0, 4] and [6, 10].
    fake = FakeFfmpeg(has_audio=True, duration=10.0, silences=[(4.0, 6.0)])
    fake.install(monkeypatch)

    video = tmp_path / "long_interview.mp4"
    video.write_bytes(b"fake-mp4")

    transcriber = FlakyTranscriber(fail_once_indices={1})
    worker = make_worker(tmp_path, transcriber, fake)
    job_id = worker.enqueue_media(video, job_id="resume-job")

    # First attempt: segment 0 succeeds, segment 1 crashes.
    first = worker.run_pending()
    assert len(first) == 1
    job = worker.get_status(job_id)
    assert job["status"] == PENDING  # requeued for retry, not lost
    assert "resume from checkpoint" in job["error"]

    job_dir = Path(worker.work_dir) / mt.file_sha256(video)[:16]
    checkpoint = json.loads((job_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["source_hash"] == mt.file_sha256(video)
    assert checkpoint["results"]["0"]["status"] == "done"
    assert checkpoint["results"]["1"]["status"] == "failed"
    assert len(checkpoint["bounds"]) == 2

    # No partial transcript is allowed to reach the scorer.
    assert not (job_dir / "transcript.txt").exists()
    assert not (job_dir / "transcript.json").exists()
    # Scratch audio from the crashed attempt is cleaned.
    assert not (job_dir / "normalized.wav").exists()
    assert not (job_dir / "segments").exists()

    # Retry: only the failed segment index reaches the model again.
    worker.run_pending()
    job = worker.get_status(job_id)
    assert job["status"] == COMPLETED, job["error"]
    assert transcriber.calls == [0, 1, 1]

    checkpoint = json.loads((job_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert all(
        checkpoint["results"][str(i)]["status"] == "done" for i in range(2)
    )
    transcript_lines = Path(job["result"]["transcript_txt"]).read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(transcript_lines) == 2
    assert "hello segment 0" in transcript_lines[0]
    assert "hello segment 1" in transcript_lines[1]

    # Normalised wav regenerated during retry then cleaned again.
    assert fake.normalize_calls == 2
    assert not (job_dir / "normalized.wav").exists()
    assert not (job_dir / "worker.lock").exists()


def test_segment_transcription_error_carries_failed_indices(tmp_path, monkeypatch):
    fake = FakeFfmpeg(has_audio=True, duration=10.0, silences=[(4.0, 6.0)])
    fake.install(monkeypatch)
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"x")

    transcriber = FlakyTranscriber(fail_once_indices={0, 1})
    with pytest.raises(SegmentTranscriptionError) as exc_info:
        MediaTranscriber(
            tmp_path / "work",
            transcriber=transcriber,
            ffmpeg="C:/fakebin/ffmpeg.exe",
            ffprobe="C:/fakebin/ffprobe.exe",
        ).transcribe(video)
    assert set(exc_info.value.failed_indices) == {0, 1}


# ---------------------------------------------------------------------------
# Restart recovery: interrupted jobs and orphaned scratch files
# ---------------------------------------------------------------------------


def test_restart_requeues_running_jobs_and_reaps_orphans(tmp_path, monkeypatch):
    fake = FakeFfmpeg()
    fake.install(monkeypatch)

    store_path = tmp_path / "jobs" / "job_store.json"
    work_dir = tmp_path / "media_work"
    store = JobStore(store_path)
    job_id = store.create_job(MEDIA_JOB, payload={"media_path": "x.mp4"})
    store.mark_running(job_id)

    # Simulate scratch files + a lock owned by a process that no longer exists.
    orphan_dir = work_dir / "deadbeefdeadbeef"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "normalized.wav").write_bytes(b"FAKE-PCM")
    (orphan_dir / "segments").mkdir()
    (orphan_dir / "segments" / "seg_00000.wav").write_bytes(b"FAKE-PCM")
    (orphan_dir / "worker.lock").write_text(
        json.dumps({"pid": 7_654_321, "temp_files": []}), encoding="utf-8"
    )

    # A live owner's lock must be left alone.
    live_dir = work_dir / "cafebabecafebabe"
    live_dir.mkdir(parents=True)
    (live_dir / "normalized.wav").write_bytes(b"FAKE-PCM")
    (live_dir / "worker.lock").write_text(
        json.dumps({"pid": os.getpid(), "temp_files": []}),
        encoding="utf-8",
    )

    worker = UploadWorker(
        store_path=store_path,
        work_dir=work_dir,
        transcriber=FlakyTranscriber(),
        scorer=fake_scorer,
    )

    assert job_id in worker.requeued
    assert JobStore(store_path).get(job_id)["status"] == PENDING
    assert orphan_dir in worker.reaped_dirs
    assert not (orphan_dir / "normalized.wav").exists()
    assert not (orphan_dir / "segments").exists()
    assert not (orphan_dir / "worker.lock").exists()
    assert (live_dir / "normalized.wav").exists()


# ---------------------------------------------------------------------------
# Text path regression: the original text scoring flow must be unaffected
# ---------------------------------------------------------------------------


def test_text_job_still_scores_without_media(tmp_path):
    worker = UploadWorker(
        store_path=tmp_path / "jobs.json",
        work_dir=tmp_path / "work",
        transcriber=FlakyTranscriber(),
        scorer=fake_scorer,
    )
    job_id = worker.enqueue_text("0.0-2.0 SPEAKER_00: this is a plain text upload")
    worker.run_pending()

    job = worker.get_status(job_id)
    assert job["status"] == COMPLETED
    assert job["type"] == TEXT_JOB
    findings = json.loads(Path(job["result"]["fallacy_analysis"]).read_text())
    assert findings[0]["text_segment"] == "this is a plain text upload"

    with pytest.raises(ValueError):
        worker.enqueue_text("   ")
