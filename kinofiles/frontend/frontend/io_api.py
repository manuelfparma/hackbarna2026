"""Backend API mounted into the Reflex server, so everything lives on :8000.

Two routes: `/api/voice` turns a recorded clip into text with the STT
subagent, and `/api/agent/chat` (from `api.py`) drives the orchestrator and
returns each reply with its narration. Reflex mounts this via
`api_transformer`, so there is no second server process to run.

Both handlers do blocking network I/O (SLNG, Nebius), so they hand that
work to a thread rather than stalling the event loop the websockets share.
"""

import asyncio
import logging
import sys
from pathlib import Path

# `agent` lives at the repo root (kinofiles/), two levels above this file
# (kinofiles/frontend/frontend/io_api.py), which isn't on sys.path when
# Reflex runs from kinofiles/frontend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile

from agent.io.stt import STT

from .api import router as agent_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

stt = STT()

api = FastAPI(title="KinoFiles API")
api.include_router(agent_router)


@api.post("/api/voice")
async def voice(audio: UploadFile) -> dict:
    """Accept a multipart-uploaded audio clip and return its transcript."""
    clip = await audio.read()
    text = await asyncio.to_thread(stt.transcribe, clip, audio.filename or "clip.webm")
    return {"text": text}
