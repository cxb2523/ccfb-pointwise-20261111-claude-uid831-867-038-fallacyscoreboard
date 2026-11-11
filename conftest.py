"""Pytest bootstrap: make the repo root importable without packaging.

Also guards against accidentally importing the heavyweight fallacyscoreboard
module (it reads config.json and imports whisperx at import time); the tests
only exercise media_transcribe / job_store / upload_worker with injected fakes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
