from fastapi import APIRouter
from pydantic import BaseModel
import sys
import os

# Add kinofiles directory to sys.path so we can import agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from agent.orchestrator import OrquestratorAgent
from langgraph.types import Command

router = APIRouter(prefix="/api/agent", tags=["agent"])

# Instantiate a global orchestrator agent
# The InMemorySaver will persist state across requests for the same thread_id
agent_instance = OrquestratorAgent()

class ChatRequest(BaseModel):
    thread_id: str
    message: str | None = None

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
            # The agent is asking a question or returning choices
            reply_val = event["__interrupt__"][0].value
            
            # Format the output if it's a dict (e.g. from the review step)
            if isinstance(reply_val, dict):
                movies = reply_val.get("movies", [])
                question = reply_val.get("question", "")
                
                reply_text = question + "\n\n"
                for i, m in enumerate(movies):
                    reply_text += f"{i+1}. {m}\n"
                    
                return {"status": "waiting_for_input", "reply": reply_text, "movies": movies}
            
            return {"status": "waiting_for_input", "reply": str(reply_val)}
        
        if "farewell" in event:
            return {"status": "done", "reply": event["farewell"], "choice": event.get("choice")}
            
        return {"status": "unknown", "reply": "An unexpected error occurred."}
        
    except Exception as e:
        return {"status": "error", "reply": str(e)}
