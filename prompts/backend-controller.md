You are an expert Python Backend Developer specializing in FastAPI and building robust APIs for AI applications.

Your goal is to generate a clean, production-ready FastAPI controller for a Retrieval-Augmented Generation (RAG) system.

---

### TASK

Build a FastAPI application that acts as the orchestration layer between a frontend and a RAG query engine.

The output should include:
1.  **FastAPI Setup**: Initialize the app with a descriptive title.
2.  **CORS Configuration**: Allow all origins, methods, and headers (for development flexibility).
3.  **Data Models**: Use Pydantic to define request and response schemas.
4.  **Endpoints**:
    *   `GET /health`: A simple health check returning `{"status": "ok"}`.
    *   `POST /ask`: The primary RAG endpoint that accepts a question and returns an AI-generated answer.
5.  **Integration**: Import and call the `answer_question` function from `rag_query.py`.
6.  **Error Handling**: Implement proper validation and exception handling with appropriate HTTP status codes.
7.  **Server Startup**: Include the `uvicorn` runner configuration.

---

### TECHNICAL REQUIREMENTS

1.  **Library Imports**:
    *   `FastAPI`, `HTTPException` from `fastapi`.
    *   `CORSMiddleware` from `fastapi.middleware.cors`.
    *   `BaseModel` from `pydantic`.
    *   Import `answer_question` from `rag_query`.

2.  **Request Schema (`AskRequest`)**:
    *   `question` (string): The user's query.

3.  **Response Schema (`AskResponse`)**:
    *   `answer` (string): The generated response from the RAG engine.

4.  **Logic for `/ask`**:
    *   Validate that the question is not an empty or whitespace-only string. Return a `400 Bad Request` if invalid.
    *   Call `answer_question(request.question)` within a try-except block.
    *   If an exception occurs during RAG processing, return a `500 Internal Server Error` with the error details.

---

### OUTPUT STYLE

*   Write clean, PEP 8 compliant Python code.
*   Include descriptive docstrings for the app and endpoints.
*   Use type hints for all parameters and return values.
*   Ensure the code is a "drop-in" replacement for `main.py`.

---

Now, generate the code for `backend/main.py` based on these instructions.
