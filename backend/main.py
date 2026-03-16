import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_query import answer_question

app = FastAPI(
    title="Ragfolio RAG API",
    description="An orchestration layer for querying resume data using RAG.",
    version="1.0.0",
)

# CORS Configuration for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """
    Schema for the RAG query request.
    """
    question: str


class AskResponse(BaseModel):
    """
    Schema for the RAG query response.
    """
    answer: str


@app.get("/health")
async def health():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Primary RAG endpoint that takes a user question, retrieves context,
    and returns an AI-generated answer.
    """
    # Validate that the question is not empty or whitespace-only
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        # Integrate with the RAG query engine
        answer = answer_question(request.question)
        return AskResponse(answer=answer)
    except Exception as e:
        # Proper exception handling with error details
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during RAG processing: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
