# ----- FastAPI Application Entry Point @ backend/main.py ------

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.src.agent import SQLAgentGenerator
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.utils.logger import get_logger

logger = get_logger(__name__)
agent_instance: SQLAgentGenerator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    
    This function initializes the heavyweight `SQLAgentGenerator` (which sets up 
    DB connections, LLM clients, and Vector Stores) exactly once when the 
    application starts, preventing overhead on every request.
    """
    global agent_instance
    try:
        logger.info("Initializing SQL Agent...")
        # agent_instance = SQLAgentGenerator(google_provider=True, bedrock_provider=False)
        agent_instance=SQLAgentGenerator()
        logger.info("SQL Agent ready.")
    except Exception as e:
        logger.error(f"Failed to initialize Agent: {e}")
        raise e
    
    yield

app = FastAPI(
    title="Caliper NLP-to-SQL API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        dict: Status of the API and the agent initialization state.
    """
    if agent_instance:
        return {"status": "healthy", "agent": "loaded"}
    return {"status": "unhealthy", "agent": "not loaded"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for processing natural language queries.
    
    Args:
        request (ChatRequest): The request body containing the user query and session ID.

    Returns:
        ChatResponse: The agent's natural language response based on SQL execution.

    Raises:
        HTTPException: 503 if agent is not loaded, 500 for internal processing errors.
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        logger.info(f"Session: {request.session_id} | Query: {request.query}")
        
        # Pass the session_id to the agent for thread-level persistence
        result = agent_instance.run(request.query, session_id=request.session_id)
        
        return ChatResponse(response=result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)