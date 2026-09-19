from fastapi import APIRouter
from pydantic import BaseModel
import base64
import sys
import os

# Add kinofiles directory to sys.path so we can import agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dotenv import load_dotenv

# orchestrator.py only loads .env under __main__, so imported here the agent
# would build its LLM client without NEBIUS_API_KEY and fail on every call.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from agent.orchestrator import OrquestratorAgent
from agent.io.tts import TTS
from langgraph.types import Command

router = APIRouter(prefix="/api/agent", tags=["agent"])

# Instantiate a global orchestrator agent
# The InMemorySaver will persist state across requests for the same thread_id
agent_instance = OrquestratorAgent()
tts = TTS()

class ChatRequest(BaseModel):
    thread_id: str
    message: str | None = None


def _reply(status: str, text: str, options: list[str] | None = None, **extra) -> dict:
    """Build a chat response, synthesizing speech for `text` alongside it.

    Only `text` is narrated: `options` are for the screen, so they never
    reach TTS. This is the single point every turn's reply passes through,
    so audio is always generated here rather than the frontend fetching it
    separately. TTS failures degrade to text-only rather than failing the
    whole turn.
    """
    payload = {"status": status, "reply": text, "options": options or [], **extra}
    if text.strip():
        try:
            payload["audio"] = base64.b64encode(tts.synthesize(text)).decode()
        except Exception:
            payload["audio"] = None
    return payload

@router.post("/chat")
def chat_with_agent(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        if req.message is None:
            # Initial invocation
            event = agent_instance.graph.invoke({}, config)
        else:
            # Resume with user input
            event = agent_instance.graph.invoke(Command(resume=req.message), config)

        if "__interrupt__" in event:
            # The agent is asking a question, optionally offering choices.
            pending = event["__interrupt__"][0].value
            return _reply(
                "waiting_for_input",
                pending["text"],
                options=pending["options"],
                participant=pending.get("participant"),
            )

        if "farewell" in event:
            return _reply("done", event["farewell"], choice=event.get("choice"))

        return {"status": "unknown", "reply": "An unexpected error occurred."}

    except Exception as e:
        return {"status": "error", "reply": str(e)}
