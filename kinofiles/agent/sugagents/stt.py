"""Speech-to-text subagent: batch transcription via SLNG's gateway.

Wraps SLNG's Deepgram Nova-3 HTTP endpoint (record-and-send, not
streaming). This is the multilingual model — "multi" auto-detects the
spoken language, so it works whether users talk in Spanish or English.
Swap `ENDPOINT` for another SLNG-hosted model later — callers only see
`transcribe`.
"""

import os
from pathlib import Path

import requests

ENDPOINT = "https://us-east.api.slng.ai/v1/stt/deepgram/nova:3"

# Maps file extension -> SLNG `encoding` value. linear16 (wav) is the only
# one the API defaults to; every other format must be named explicitly.
ENCODING_BY_SUFFIX = {
    ".wav": "linear16",
    ".flac": "flac",
    ".mp3": "mp3",
    ".mp4": "mp4",
    ".m4a": "mp4",
    ".webm": "webm",
    ".aac": "aac",
    ".ogg": "ogg",
    ".opus": "opus",
}


class STT:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["SLNG_API_KEY"]

    def transcribe(self, audio: bytes, filename: str = "audio.wav", language: str = "multi") -> str:
        """Send a full audio clip to SLNG and return the transcript text."""
        encoding = ENCODING_BY_SUFFIX.get(Path(filename).suffix.lower())
        data = {"language": language, "punctuate": "true"}
        if encoding:
            data["encoding"] = encoding

        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"audio": (filename, audio)},
            data=data,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    path = sys.argv[1]
    with open(path, "rb") as f:
        print(STT().transcribe(f.read(), filename=path))
