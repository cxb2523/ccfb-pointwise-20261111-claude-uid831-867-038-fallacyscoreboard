"""Media transcription pipeline for FallacyScoreboard.

The upload entry point used to accept plain text only; this module closes the
gap for media uploads (mp4/mov/mkv/mp3/wav):

1. Locate ffmpeg/ffprobe via :func:`shutil.which` (clear install hint if missing).
2. Probe the container and reject files without an audio track.
3. Normalise the audio track to 16 kHz mono PCM wav.
4. Split the wav on silence and transcribe segment by segment (never the whole
   file at once), mapping every segment timestamp back to the original timeline.
5. Persist progress per segment in a checkpoint keyed by the source file hash
   plus the segment index, so a worker retry only re-runs the failed segments.
6. Reject pure-music / long-silence uploads explicitly instead of feeding
   empty text downstream (which would silently produce a zero fallacy score).
The transcriber is injected as a callable; the default WhisperX backend is
imported lazily so importing this module never requires torch/whisperx (and the
test-suite can run with a lightweight fake).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MediaTranscriptionError(Exception):
    """Base class for every media-pipeline failure."""


class FfmpegNotFoundError(MediaTranscriptionError):
    """Raised when the ffmpeg / ffprobe binaries cannot be located."""


class NoAudioTrackError(MediaTranscriptionError):
    """Raised when the uploaded container has no audio stream."""


class NoSpeechDetectedError(MediaTranscriptionError):
    """Raised for long-silence / pure-molicy uploads with no usable speech."""


class MediaProbeError(MediaTranscriptionError):
    """Raised when ffprobe/ffmpeg cannot inspect the source file."""


class SegmentTranscriptionError(MediaTranscriptionError):
    """Raised when one or more segments could not be transcribed.

    The checkpoint on disk still records the completed segments, so retrying
    the job only re-runs ``failed_indices``.
    """

    def __init__(self, failed_indices, *args):
        self.failed_indices = list(failed_indices)
        super().__init__(
            f"transcription failed for segment(s): {self.failed_indices}",
            *args,
        )


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".mp3", ".wav"}

# Silence detection defaults (tuned for spoken dialogue).
SILENCE_DB = "-35dB"
MIN_SILENCE_SECONDS = 2.0
MAX_SEGMENT_SECONDS = 30.0
MIN_SPEECH_SECONDS = 0.3
# A segment whose recognised text is shorter than this counts as empty.
MIN_TEXT_LENGTH = 1

_FFMPEG_INSTALL_HINT = (
    "ffmpeg is required for media transcription. Install it and make sure "
    "'ffmpeg' and 'ffprobe' are on PATH:\n"
    "  Windows (winget):  winget install Gyan.FFmpeg\n"
    "  macOS (Homebrew):  brew install ffmpeg\n"
    "  Debian/Ubuntu:      sudo apt-get install ffmpeg\n"
    "Then restart the upload worker."
)


# ---------------------------------------------------------------------------
# Process / binary seams (patched by the test-suite)
# ---------------------------------------------------------------------------


def _which(binary):
    """Thin wrapper around :func:`shutil.which` so tests can stub lookup."""
    return shutil.which(binary)


def _run_command(cmd):
    """Run a command, capturing output. Raises with stderr on failure."""
    proc = subprocess.run(
        [str(part) for part in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"{proc.stderr.strip()}"
        )
    return proc


def ensure_ffmpeg():
    """Return ``(ffmpeg_path, ffprobe_path)`` or raise with an install hint."""
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise FfmpegNotFoundError(_FFMPEG_INSTALL_HINT)
    # ffprobe ships alongside ffmpeg in every standard build; fall back to
    # parsing `ffmpeg -i` stderr when a distribution omits it.
    ffprobe = _which("ffprobe")
    return ffmpeg, ffprobe


# ---------------------------------------------------------------------------
# Hashing / probing / normalisation
# ---------------------------------------------------------------------------


def file_sha256(path, chunk_size=1024 * 1024):
    """Stream-hash a source file so checkpoints are content-addressed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_media(path, ffprobe=None, ffmpeg=None):
    """Inspect a media file.

    Returns a dict ``{"duration": float, "has_audio": bool, "streams": [...]}``.
    Uses ffprobe when available and otherwise parses ``ffmpeg -i`` stderr.
    """
    if ffprobe:
        return _probe_with_ffprobe(path, ffprobe)
    return _probe_with_ffmpeg(path, ffmpeg or _which("ffmpeg"))


def _probe_with_ffprobe(path, ffprobe):
    try:
        proc = _run_command(
            [
                ffprobe, "-v", "error",
                "-print_format", "json",
                "-show_format", "-show_streams",
                path,
            ]
        )
    except RuntimeError as exc:
        raise MediaProbeError(f"ffprobe failed for {path}: {exc}") from exc

    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"unparseable ffprobe output for {path}") from exc

    streams = meta.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    try:
        duration = float(meta.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration": duration, "has_audio": has_audio, "streams": streams}


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_AUDIO_LINE_RE = re.compile(r"Stream #\d+[:\d]*.*?: Audio:")


def _probe_with_ffmpeg(path, ffmpeg):
    if not ffmpeg:
        raise FfmpegNotFoundError(_FFMPEG_INSTALL_HINT)
    # `ffmpeg -i` without an output always exits non-zero; call subprocess
    # directly instead of _run_command.
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr = proc.stderr
    if "Invalid data found" in stderr or "Error opening" in stderr:
        raise MediaProbeError(f"ffmpeg could not open {path}: {stderr.strip()}")
    has_audio = bool(_AUDIO_LINE_RE.search(stderr))
    duration = 0.0
    match = _DURATION_RE.search(stderr)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return {"duration": duration, "has_audio": has_audio, "streams": []}


def normalize_audio(source_path, wav_path, ffmpeg):
    """Extract/convert the audio track to 16 kHz mono 16-bit PCM wav."""
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_command(
            [
                ffmpeg, "-y",
                "-i", str(source_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ]
        )
    except RuntimeError as exc:
        raise MediaTranscriptionError(
            f"failed to normalise audio from {source_path}: {exc}"
        ) from exc
    if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
        raise MediaTranscriptionError(
            f"ffmpeg produced no audio data for {source_path}"
        )


# ---------------------------------------------------------------------------
# Silence-based segmentation
# ---------------------------------------------------------------------------


_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(\d+(?:\.\d+)?)")


def detect_silence(wav_path, ffmpeg, noise=SILENCE_DB, min_duration=MIN_SILENCE_SECONDS):
    """Return silence intervals as a list of ``(start, end)`` second pairs."""
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner",
            "-i", str(wav_path),
            "-af", f"silencedetect=noise={noise}:d={min_duration}",
            "-f", "null", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # silencedetect writes to stderr and ffmpeg exits non-zero only on real errors
    if proc.returncode != 0 and "silence_" not in proc.stderr:
        raise MediaTranscriptionError(
            f"silence detection failed for {wav_path}: {proc.stderr.strip()}"
        )
    intervals = []
    starts = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(proc.stderr)]
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(proc.stderr)]
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else None
        if end is not None:
            intervals.append((max(0.0, start), end))
        # A trailing silence_start without silence_end runs to EOF; the
        # segment builder handles the open end via the known duration.
    return intervals


def _clamp_span(start, end, max_len, out):
    """Split a speech span longer than ``max_len`` into bounded chunks."""
    cursor = start
    while end - cursor > max_len:
        out.append((round(cursor, 3), round(cursor + max_len, 3)))
        cursor += max_len
    if end - cursor >= MIN_SPEECH_SECONDS:
        out.append((round(cursor, 3), round(end, 3)))


def build_speech_segments(duration, silences, max_len=MAX_SEGMENT_SECONDS):
    """Turn silence intervals into speech segments on the original timeline."""
    segments = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        silence_start = min(silence_start, duration)
        if silence_start - cursor >= MIN_SPEECH_SECONDS:
            _clamp_span(cursor, silence_start, max_len, segments)
        cursor = max(cursor, min(silence_end, duration))
    if duration - cursor >= MIN_SPEECH_SECONDS:
        _clamp_span(cursor, duration, max_len, segments)
    return segments


def cut_segment(ffmpeg, wav_path, start, end, segment_path):
    """Cut one speech segment out of the normalised wav."""
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            ffmpeg, "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{max(0.0, end - start):.3f}",
            "-i", str(wav_path),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(segment_path),
        ]
    )


# ---------------------------------------------------------------------------
# Default WhisperX transcriber (lazy)
# ---------------------------------------------------------------------------


def build_whisperx_transcriber(model_name="base", device=None):
    """Build a transcriber callable backed by WhisperX.

    Imported lazily: worker machines without torch can still import this module
    (e.g. to run the text-only path or the test-suite).
    """
    import torch  # type: ignore
    import whisperx  # type: ignore

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = whisperx.load_model(model_name, resolved_device)

    def transcribe(segment_wav_path):
        audio = whisperx.load_audio(str(segment_wav_path))
        result = model.transcribe(audio)
        return {
            "text": result.get("text", ""),
            "segments": [
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", "").strip(),
                }
                for seg in result.get("segments", [])
            ],
        }

    return transcribe


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def format_clock(seconds):
    """Render a second offset as ``mm:ss.s`` so findings map to分秒."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:04.1f}"


def _map_to_original(local_result, global_start, global_end):
    """Map a segment-local recognition result onto the original timeline."""
    local_segments = local_result.get("segments") or []
    mapped = []
    for seg in local_segments:
        local_start = float(seg.get("start", 0.0))
        local_end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        mapped.append(
            {
                "start": round(global_start + local_start, 3),
                "end": round(global_start + local_end, 3),
                "text": text,
            }
        )
    if not mapped:
        text = (local_result.get("text") or "").strip()
        if text:
            mapped.append(
                {"start": round(global_start, 3), "end": round(global_end, 3), "text": text}
            )
    return mapped


# ---------------------------------------------------------------------------
# Lock / orphan cleanup
# ---------------------------------------------------------------------------


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is not ours (Windows signals this way).
        return True
    except OSError:
        return False
    return True


def write_lock(lock_path, temp_files):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "temp_files": [str(p) for p in temp_files]}
    tmp = lock_path.with_suffix(".lock.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, lock_path)


def cleanup_temp_files(job_dir):
    """Remove the normalised wav and the segment scratch directory."""
    removed = []
    normalized = Path(job_dir) / "normalized.wav"
    if normalized.exists():
        normalized.unlink()
        removed.append(normalized)
    segments_dir = Path(job_dir) / "segments"
    if segments_dir.exists():
        shutil.rmtree(segments_dir, ignore_errors=True)
        removed.append(segments_dir)
    return removed


def release_lock(lock_path):
    try:
        Path(lock_path).unlink()
    except FileNotFoundError:
        pass


def reap_orphans(base_dir):
    """Clean temp files / locks left by workers that died or were restarted.

    A job directory is considered orphaned when its ``worker.lock`` references
    a PID that no longer exists. Completed jobs never hold a lock, and the
    checkpoint/transcript outputs are deliberately kept for resume/auditing.
    """
    base = Path(base_dir)
    reaped = []
    if not base.exists():
        return reaped
    for lock_path in base.glob("*/worker.lock"):
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"pid": None, "temp_files": []}
        if _pid_alive(payload.get("pid")):
            continue
        job_dir = lock_path.parent
        cleanup_temp_files(job_dir)
        release_lock(lock_path)
        reaped.append(job_dir)
    return reaped


# ---------------------------------------------------------------------------
# Checkpointed pipeline
# ---------------------------------------------------------------------------


class MediaTranscriber:
    """Runs the full segment-by-segment pipeline with hash-based checkpoints.

    Parameters
    ----------
    work_dir:
        Root directory holding per-job scratch directories.
    transcriber:
        Callable ``(segment_wav_path) -> {"text", "segments": [...]}``. When
        ``None`` a WhisperX ``base`` model is built on first use.
    ffmpeg / ffprobe:
        Explicit binary paths; discovered via :func:`shutil.which` otherwise.
    """

    def __init__(self, work_dir, transcriber=None, ffmpeg=None, ffprobe=None):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._transcriber = transcriber
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._lazy_model = None

    # -- transcriber handling ------------------------------------------------

    def _get_transcriber(self):
        if self._transcriber is not None:
            return self._transcriber
        if self._lazy_model is None:
            self._lazy_model = build_whisperx_transcriber()
        return self._lazy_model

    # -- checkpoint IO --------------------------------------------------------

    @staticmethod
    def _load_checkpoint(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_checkpoint_atomic(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".checkpoint.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)

    # -- main entry point -----------------------------------------------------

    def transcribe(self, source_path, progress_callback=None):
        """Transcribe one media upload; returns a result dict.

        Returns
        -------
        dict with keys:
            ``source_hash``, ``job_dir``, ``duration``,
            ``segments`` (mapped to the original timeline with clock strings),
            ``transcript_json``, ``transcript_txt``, ``checkpoint``.
        """
        source_path = Path(source_path)
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise MediaTranscriptionError(
                f"unsupported media extension '{source_path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        ffmpeg, ffprobe = ensure_ffmpeg()
        self._ffmpeg = self._ffmpeg or ffmpeg
        self._ffprobe = self._ffprobe or ffprobe

        source_hash = file_sha256(source_path)
        job_dir = self.work_dir / source_hash[:16]
        checkpoint_path = job_dir / "checkpoint.json"
        normalized = job_dir / "normalized.wav"
        segments_dir = job_dir / "segments"
        lock_path = job_dir / "worker.lock"
        transcript_json = job_dir / "transcript.json"
        transcript_txt = job_dir / "transcript.txt"

        job_dir.mkdir(parents=True, exist_ok=True)
        write_lock(lock_path, [normalized, segments_dir])

        try:
            probe = probe_media(source_path, self._ffprobe, self._ffmpeg)
            if not probe["has_audio"]:
                raise NoAudioTrackError(
                    f"{source_path.name} contains no audio track; nothing to "
                    "transcribe. Upload a media file with dialogue audio."
                )
            duration = probe["duration"]

            # 1. Normalise (idempotent across retries).
            if not normalized.exists():
                normalize_audio(source_path, normalized, self._ffmpeg)

            # 2. Silence-based segmentation.
            silences = detect_silence(normalized, self._ffmpeg)
            bounds = build_speech_segments(duration, silences)
            if not bounds:
                raise NoSpeechDetectedError(
                    f"{source_path.name} appears to be long silence or music "
                    "without speech; no transcribable dialogue was found."
                )

            # 3. Resume only when the checkpoint matches this exact content
            #    and segmentation; otherwise start the segment run afresh.
            checkpoint = self._load_checkpoint(checkpoint_path)
            expected_bounds = [[a, b] for a, b in bounds]
            if (
                not checkpoint
                or checkpoint.get("source_hash") != source_hash
                or checkpoint.get("bounds") != expected_bounds
            ):
                checkpoint = {
                    "source_hash": source_hash,
                    "source_path": str(source_path),
                    "duration": duration,
                    "bounds": expected_bounds,
                    "results": {},
                }
                self._write_checkpoint_atomic(checkpoint_path, checkpoint)

            # 4. Cut the segment wavs (skip segments already on disk).
            segment_files = []
            for index, (start, end) in enumerate(bounds):
                seg_path = segments_dir / f"seg_{index:05d}.wav"
                segment_files.append(seg_path)
                if not seg_path.exists():
                    cut_segment(self._ffmpeg, normalized, start, end, seg_path)

            # 5. Transcribe segment by segment; only failed/missing segments run.
            transcriber = self._get_transcriber()
            failed_indices = []
            for index, (start, end) in enumerate(bounds):
                key = str(index)
                saved = checkpoint["results"].get(key)
                if saved and saved.get("status") == "done":
                    continue
                try:
                    local_result = transcriber(str(segment_files[index]))
                    mapped = _map_to_original(local_result, start, end)
                    checkpoint["results"][key] = {
                        "status": "done",
                        "mapped": mapped,
                    }
                except Exception as exc:  # recorded; the segment is retried later
                    checkpoint["results"][key] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failed_indices.append(index)
                self._write_checkpoint_atomic(checkpoint_path, checkpoint)
                if progress_callback:
                    progress_callback(index + 1, len(bounds))

            if failed_indices:
                raise SegmentTranscriptionError(failed_indices)

            # 6. Flatten the checkpoint into timestamped transcript outputs.
            merged = []
            for index in range(len(bounds)):
                merged.extend(checkpoint["results"][str(index)]["mapped"])
            merged.sort(key=lambda item: item["start"])

            if not any(len(item["text"]) >= MIN_TEXT_LENGTH for item in merged):
                raise NoSpeechDetectedError(
                    f"{source_path.name} yielded no recognised speech "
                    "(pure music or long silence); refusing to score empty text."
                )

            transcript_payload = {
                "source_hash": source_hash,
                "source_path": str(source_path),
                "duration": duration,
                "segments": [
                    {
                        "index": idx,
                        "start": item["start"],
                        "end": item["end"],
                        "start_clock": format_clock(item["start"]),
                        "end_clock": format_clock(item["end"]),
                        "text": item["text"],
                    }
                    for idx, item in enumerate(merged)
                ],
            }
            tmp_json = transcript_json.with_suffix(".json.tmp")
            with open(tmp_json, "w", encoding="utf-8") as handle:
                json.dump(transcript_payload, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_json, transcript_json)

            # Line format consumed by llm.detect_fallacies / prompt_ollama:
            # "{start}-{end} {speaker}: {text}"
            lines = [
                f"{item['start']:.3f}-{item['end']:.3f} SPEAKER_00: {item['text']}"
                for item in merged
            ]
            tmp_txt = transcript_txt.with_suffix(".txt.tmp")
            with open(tmp_txt, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            os.replace(tmp_txt, transcript_txt)

            return {
                "source_hash": source_hash,
                "job_dir": str(job_dir),
                "duration": duration,
                "segments": transcript_payload["segments"],
                "transcript_json": str(transcript_json),
                "transcript_txt": str(transcript_txt),
                "checkpoint": str(checkpoint_path),
            }
        finally:
            # Normal completion and exception paths alike drop the scratch
            # audio; the checkpoint and transcripts remain for resume/audit.
            cleanup_temp_files(job_dir)
            release_lock(lock_path)
