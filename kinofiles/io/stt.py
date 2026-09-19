"""Speech-to-text subagent: batch transcription via SLNG's gateway.

Wraps SLNG's Deepgram Nova-3 HTTP endpoint (record-and-send, not
streaming). This is the multilingual model — "multi" auto-detects the
spoken language, so it works whether users talk in Spanish or English.
Swap `ENDPOINT` for another SLNG-hosted model later — callers only see
`transcribe`.
"""

import os

import requests

ENDPOINT = "https://us-east.api.slng.ai/v1/stt/deepgram/nova:3"


class STT:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["SLNG_API_KEY"]

    def transcribe(self, audio: bytes, filename: str = "audio.wav", language: str = "multi") -> str:
        """Send a full audio clip to SLNG and return the transcript text."""
        # No `encoding`: that parameter is for raw headerless audio and makes
        # the API reject containers like webm. Formats are auto-detected.
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"audio": (filename, audio)},
            data={"language": language, "punctuate": "true"},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"SLNG {response.status_code}: {response.text}")
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    path = sys.argv[1]
    with open(path, "rb") as f:
        print(STT().transcribe(f.read(), filename=path))
