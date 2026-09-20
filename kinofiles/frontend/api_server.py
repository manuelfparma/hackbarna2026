import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frontend.api import router as agent_router

app = FastAPI(title="KinoFiles Agent API")

# Allow CORS for the Reflex frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    # The Hub relays server-side notices (e.g. the unauthenticated-request
    # nag) through this logger, which also carries its own handler, so the
    # same line lands twice. Silencing the logger suppresses both copies.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
