# 01_ollama.py
# Optional: start `ollama serve` in the background (same idea as course labs).
# From repo root: python 01_ollama.py

import os
import subprocess
import time

PORT = 11434
OLLAMA_HOST = f"0.0.0.0:{PORT}"
OLLAMA_CONTEXT_LENGTH = 32000

os.environ["OLLAMA_HOST"] = OLLAMA_HOST
os.environ["OLLAMA_CONTEXT_LENGTH"] = str(OLLAMA_CONTEXT_LENGTH)

process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(1)
