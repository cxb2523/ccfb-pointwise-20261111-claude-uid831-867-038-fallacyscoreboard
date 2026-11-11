# FallacyScoreboard

FallacyScoreboard is an offline tool that downloads YouTube videos, processes them using WhisperX for automatic speech recognition and speaker diarization, detects logical fallacies using LLMs with OLLAMA, and overlays the results on the video with a scoreboard.

## Demo Video

[![FallacyScoreboard Demo](https://img.youtube.com/vi/1plEKbSHJMM/0.jpg)](https://youtu.be/1plEKbSHJMM)

Click the image above to watch a demonstration of FallacyScoreboard in action!

## Features

- Download YouTube videos and extract audio
- Transcribe audio using WhisperX with word-level timestamps and speaker diarization
- Detect logical fallacies in transcribed text
- Overlay fallacy detection results on the video with a scoreboard and side panel
- Handle multiple fallacies per speaker segment
- Color-code speakers for easy identification

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/latent-variable/FallacyScoreboard.git
    cd FallacyScoreboard
    ```

2. **Create a Python environment**:
    ```bash
    conda create --name fallacyscoreboard python=3.10
    conda activate fallacyscoreboard
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install ffmpeg** (required by the media upload pipeline; located via
   `shutil.which`, so it must be on `PATH`):
    ```bash
    # Windows
    winget install Gyan.FFmpeg
    # macOS
    brew install ffmpeg
    # Debian/Ubuntu
    sudo apt-get install ffmpeg
    ```

5. **Install ImageMagick**:
   FallacyScoreboard uses MoviePy, which requires ImageMagick. Make sure to install ImageMagick and set the path in the script:
   ```python
   change_settings({"IMAGEMAGICK_BINARY": r"path/to/your/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})
   ```

## Usage

1. **Process a YouTube video**:
    ```python
    from fallacyscoreboard import fallacy_detection_pipeline

    output_path = "path/to/output/directory"
    video_url = "https://www.youtube.com/watch?v=example"
    
    fallacy_detection_pipeline(video_url, output_path)
    ```

   This function will:
   - Download the YouTube video
   - Extract audio from the video
   - Transcribe the audio using WhisperX
   - Detect fallacies in the transcription
   - Overlay the fallacies and scoreboard on the video

2. **Customize OLLAMA settings**:
   You can customize the OLLAMA host and model by modifying these variables in the script:
   ```python
   OLLAMA_HOST = 'http://localhost:11434'
   OLLAMA_MODEL = 'llama3.1'
   ```

3. **Queue a media upload for transcription** (mp4/mov/mkv/mp3/wav):
   ```python
   from upload_worker import UploadWorker

   worker = UploadWorker(
       store_path="Files/jobs/job_store.json",  # persistent, poll-safe job store
       work_dir="Files/media_work",             # per-job scratch + checkpoints
   )
   job_id = worker.enqueue_media("interview.mp4")
   worker.run_pending()                          # or worker.start() for a loop

   status = worker.get_status(job_id)            # safe to poll / refresh
   print(status["status"], status["result"]["transcript_txt"])
   ```

   Plain-text uploads keep using the original scoring path unchanged:
   ```python
   job_id = worker.enqueue_text("0.0-2.0 SPEAKER_00: transcribed dialogue ...")
   ```

### Media pipeline behavior

- **Probe & normalise**: ffmpeg/ffprobe are located with `shutil.which` (a
  missing binary raises with an install hint). The audio track is normalised
  to 16 kHz mono PCM wav. A container without an audio track fails with
  `NoAudioTrackError`; pure-music / long-silence audio fails with
  `NoSpeechDetectedError` — empty text is never sent to the scorer, so a
  media upload can never silently produce a zero fallacy score.
- **Segmented transcription**: the wav is split on silence and each segment is
  recognised independently. Segment-local timestamps are mapped back onto the
  original audio timeline; the transcript is written in the same
  `"{start}-{end} SPEAKER_00: {text}"` line format consumed by
  `llm.detect_fallacies`, so every detected fallacy remains locatable to the
  second (`start_clock`/`end_clock` in `transcript.json`).
- **Checkpoint & resume**: checkpoints are keyed by the source file SHA-256
  plus the segment index. A worker retry re-runs only the failed segments;
  jobs are requeued automatically until the attempt limit is reached.
- **Durable job state**: all jobs live in an atomically written JSON job
  store. Polling or refreshing never makes a task disappear, and on worker
  restart jobs left `running` are requeued while orphaned wav/segment files
  are reaped via per-job `worker.lock` files.
- **Cleanup guarantees**: the normalised wav and segment scratch directory are
  removed on both normal completion and exception paths; `reap_orphans()`
  handles files left by crashed/restarted processes. Checkpoints and
  transcripts are retained for resume and auditing.

## Output

The script will generate several files in the output directory:
- The downloaded video file
- An extracted audio file
- A JSON file with the transcription and diarization results
- A text file with formatted transcription
- A JSON file with fallacy analysis results
- The final video with overlaid fallacy scoreboard and explanations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Notes

- Ensure you have sufficient disk space and computational resources, as processing videos can be resource-intensive.
- The quality of fallacy detection depends on the OLLAMA model used. Experiment with different models for best results.
- This tool is for educational and research purposes. Always respect copyright and terms of service when using YouTube content.
