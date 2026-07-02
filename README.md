# SHL Assessment Recommender AI

A conversational AI-powered recommendation system that suggests relevant **SHL Individual Test Solutions** using **Retrieval-Augmented Generation (RAG)**, semantic search, and a decision-driven conversational workflow.

The API is designed to support recruiters by asking clarifying questions, refining recommendations as requirements evolve, comparing assessments, and maintaining conversational context in a stateless architecture.

---

## Features

- Conversational recommendation workflow
- Clarifying questions for incomplete requirements
- Context-aware multi-turn conversations
- Recommendation refinement when requirements change
- Assessment comparison using catalog evidence
- Off-topic request handling
- Retrieval-Augmented Generation (RAG)
- Semantic search using FAISS
- REST API built with FastAPI
- Public Swagger documentation

---

## Architecture

```
                   FastAPI
                      │
                      ▼
            ConversationAgent
        (Workflow Orchestrator)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
ConversationAnalyzer         SearchEngine
(Language Understanding)   (FAISS Retrieval)
        │                           │
        └─────────────┬─────────────┘
                      ▼
                 LLM (Gemini)
                      │
                      ▼
               JSON API Response
```

---

## Project Structure

```
assessment-recommender-ai/
│
├── app/
│   ├── agent/
│   │   ├── conversation_agent.py
│   │   └── conversation_analyzer.py
│   │
│   ├── core/
│   │   └── llm.py
│   │
│   ├── retrieval/
│   │   └── search_engine.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── scripts/
│   ├── preprocess_catalog.py
│   └── build_embeddings.py
│
├── requirements.txt
└── README.md
```

---

## Retrieval Pipeline

1. User conversation is received.
2. Conversation Analyzer extracts structured search context.
3. Search context is converted into a semantic search query.
4. FAISS retrieves the most relevant SHL assessments.
5. Metadata filtering and ranking improve retrieval quality.
6. Gemini generates the final conversational response grounded in retrieved catalog entries.

---

## Evaluation

The system was evaluated using representative multi-turn hiring conversations covering the required workflows:

- Clarification for incomplete hiring requirements
- Assessment recommendation
- Recommendation refinement when user constraints changed
- Assessment comparison
- Off-topic request handling

Evaluation focused on:

- Retrieval Quality – verifying that FAISS retrieved relevant SHL catalog entries.
- Recommendation Relevance – ensuring returned assessments matched the hiring requirements.
- Groundedness – confirming recommendations referenced only products present in the processed SHL catalog.
- Conversation Accuracy – validating correct workflow transitions (clarify → recommend → refine → compare).

Testing was performed using multiple conversation scenarios derived from the SHL assignment requirements.

## Conversation Workflow

The agent supports five workflows:

- Clarify
- Recommend
- Refine
- Compare
- Refuse (off-topic requests)

The API is **stateless**. Every request contains the complete conversation history, allowing the search context to be reconstructed on every turn.

---

## API Endpoints

### Health Check

```
GET /health
```

Response

```json
{
  "status": "ok"
}
```

---

### Chat Endpoint

```
POST /chat
```

Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "We are hiring a Java Developer."
    }
  ]
}
```

Response

```json
{
  "reply": "...",
  "recommendations": [
    {
      "name": "...",
      "url": "...",
      "test_type": "..."
    }
  ],
  "end_of_conversation": false
}
```

---

## Local Setup

Clone the repository

```bash
git clone <repository-url>
cd assessment-recommender-ai
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

## Prepare the Catalog

Generate the processed catalog

```bash
python scripts/preprocess_catalog.py
```

Generate embeddings and FAISS index

```bash
python scripts/build_embeddings.py
```

---

## Run the API

Development

```bash
uvicorn app.main:app --reload
```

Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Swagger documentation

```
http://localhost:8000/docs
```

---

## Deployment

The project is deployed on Railway.

Public API

```
https://assessment-recommender-ai-production.up.railway.app
```

Swagger

```
https://assessment-recommender-ai-production.up.railway.app/docs
```

Health Check

```
https://assessment-recommender-ai-production.up.railway.app/health
```

---

## Technologies Used

- Python
- FastAPI
- Google Gemini 2.5 Flash
- FAISS
- Sentence Transformers
- Hugging Face
- Pydantic
- NumPy

---

## Notes

- The API is stateless.
- Recommendations are generated only from the processed SHL catalog.
- Environment variables are managed securely and are not committed to the repository.

---

## License

This project was developed as part of the SHL AI Intern Hiring Assessment.