from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.conversation_agent import ConversationAgent


# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Conversational SHL Assessment Recommendation Service",
    version="1.0.0",
)

# Create a single agent instance
agent = ConversationAgent()


# -------------------------------------------------------
# Request Models
# -------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


# -------------------------------------------------------
# Response Models
# -------------------------------------------------------

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# -------------------------------------------------------
# Routes
# -------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommender API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        result = agent.chat(
            [message.model_dump() for message in request.messages]
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )