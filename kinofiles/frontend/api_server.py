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
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=True)
