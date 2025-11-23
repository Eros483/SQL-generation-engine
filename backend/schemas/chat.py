from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""
    query: str
class ChatResponse(BaseModel):
    """Schema for the AI response."""
    response: str
    success: bool = True