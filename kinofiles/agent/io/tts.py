"""Text-to-speech subagent: batch synthesis via SLNG's gateway.

Wraps SLNG's Deepgram Aura-2 HTTP endpoint (`aura:2-en`, which despite the
suffix serves both English and Spanish). Mirrors `stt.py`: swap `ENDPOINT`
or `DEFAULT_VOICE` for another SLNG-hosted model later — callers only see
`synthesize`.
"""

import os

import requests

ENDPOINT = "https://us-east.api.slng.ai/v1/tts/slng/deepgram/aura:2-en"
DEFAULT_VOICE = "aura-2-thalia-en"


class TTS:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["SLNG_API_KEY"]

    def synthesize(self, text: str, voice: str = DEFAULT_VOICE) -> bytes:
        """Send text to SLNG and return the synthesized audio (wav bytes)."""
        # No `encoding`/`container`: this Aura route rejects audio settings
        # ("Audio settings are not supported for this Deepgram Aura route").
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"text": text, "model": voice},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"SLNG {response.status_code}: {response.text}")
        return response.content


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello from KinoFiles."
    out_path = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
    with open(out_path, "wb") as f:
        f.write(TTS().synthesize(text))
    print(f"wrote {out_path}")
