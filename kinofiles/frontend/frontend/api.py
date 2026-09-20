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
    voice: str | None = None


def _reply(
    status: str,
    text: str,
    options: list[str] | None = None,
    voice: str | None = None,
    explanation: str = "",
    **extra,
) -> dict:
    """Build a chat response, synthesizing speech for the spoken parts.

    Only `options` are kept out of TTS: they are for the screen, and a list
    of titles makes for terrible speech. `explanation` is the mediator
    accounting for the shortlist, so it is spoken ahead of the question but
    returned as its own field, letting the frontend give it its own place on
    screen instead of running it into the question. This is the single point
    every turn's reply passes through, so audio is always generated here
    rather than the frontend fetching it separately. TTS failures degrade to
    text-only rather than failing the whole turn.
    """
    payload = {
        "status": status,
        "reply": text,
        "explanation": explanation,
        "options": options or [],
        **extra,
    }
    narration = " ".join(part for part in (explanation, text) if part.strip())
    if narration.strip():
        try:
            if voice:
                payload["audio"] = base64.b64encode(tts.synthesize(narration, voice=voice)).decode()
            else:
                payload["audio"] = base64.b64encode(tts.synthesize(narration)).decode()
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
            
            state = agent_instance.graph.get_state(config).values
            participants = state.get("participants", [])
            votes = state.get("votes", {})
            current_participant_idx = state.get("current_participant", 0)
            current_participant = pending.get("participant")
            if not current_participant and participants and current_participant_idx < len(participants):
                current_participant = participants[current_participant_idx]
                
            return _reply(
                "waiting_for_input",
                pending["text"],
                options=pending["options"],
                voice=req.voice,
                explanation=pending.get("explanation", ""),
                participant=current_participant,
                participants=[p for p in participants if p],
                votes=votes,
                criteria=pending.get("criteria", {}),
            )

        if "farewell" in event:
            state = agent_instance.graph.get_state(config).values
            return _reply(
                "done", 
                event["farewell"], 
                choice=event.get("choice"),
                voice=req.voice,
                participants=[p for p in state.get("participants", []) if p],
                votes=state.get("votes", {})
            )

        return {"status": "unknown", "reply": "An unexpected error occurred."}

    except Exception as e:
        return {"status": "error", "reply": str(e)}
