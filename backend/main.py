import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.src.agent import SQLAgentGenerator
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Global variable to store the agent instance
agent_instance: SQLAgentGenerator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    We initialize the heavyweight Agent (DB connections/LLM setup) here ONCE.
    """
    global agent_instance
    try:
        logger.info("Initializing SQL Agent...")
        agent_instance = SQLAgentGenerator()
        logger.info("SQL Agent ready.")
    except Exception as e:
        logger.error(f"Failed to initialize Agent: {e}")
        raise e
    
    yield
    
    # Cleanup code (close DB connections) could go here if your class supported it
    logger.info("Shutting down...")

app = FastAPI(
    title="Caliper NLP-to-SQL API",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
# Essential for allowing your Streamlit/React frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to ["http://localhost:8501"] for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.get("/health")
async def health_check():
    """Simple health check to ensure API is running."""
    if agent_instance:
        return {"status": "healthy", "agent": "loaded"}
    return {"status": "unhealthy", "agent": "not loaded"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint. Receives a query, runs the agent graph, returns the answer.
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        logger.info(f"Received query: {request.query}")
        
        # Run the agent (Synchronously for now, as LangGraph runs sync by default)
        result = agent_instance.run(request.query)
        
        return ChatResponse(response=result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Standard development run command
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)