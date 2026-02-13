# Building a RAG-Powered News Q&A System with FastAPI & Qdrant

Backend for saving news URLs and querying them with AI-powered search.

---

## Postman Collection

You can test all endpoints using the Postman collection below:

**Postman Collection:**  
[https://www.postman.com/myselfmdsohail-1533277/linkmind](https://www.postman.com/myselfmdsohail-1533277/linkmind)

## Running the Backend Locally

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

## Routes

| Route               | Method | Use                                                                 |
|---------------------|--------|----------------------------------------------------------------------|
| `/v1/api/save-url`  | POST   | Ingest news URLs; content is scraped, chunked, embedded, and stored in Qdrant. |
| `/v1/api/query`     | POST   | Search stored news and (with `backend: "custom"`) get an AI answer from your LLM. |

---

## Payloads

### POST `/v1/api/save-url`

```json
{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ]
}
```

---

### POST `/v1/api/query`

**Headers**

```
api_key: YOUR_API_KEY
```

**Body**

```json
{
  "query": "Your question",
  "backend": "custom",
  "model": "your-model-id",
  "custom_url": "https://your-llm-api/completions"
}
```

---

## Custom LLM API Structure Requirement

Currently, the backend supports only a custom LLM endpoint.

Your custom LLM server must accept the following request structure:

### Request Format (Sent by Backend)

```json
{
  "model": "your-model-id",
  "messages": [
    {
      "role": "system",
      "content": "System instructions with retrieved context"
    },
    {
      "role": "user",
      "content": "User query"
    }
  ],
  "max_tokens": 300
}
```

---

## Required Headers

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

---

## Response Handling

The backend does **not enforce a strict response schema** for the custom LLM.

Whatever valid JSON response is returned by the external LLM endpoint will be forwarded directly to the client without modification.

This means:

- The backend does not validate or transform the response structure.
- Any valid JSON format returned by your LLM server will be passed through as-is.

---

# Deployment Notes

## Vercel Deployment (Failed)

Deployment to Vercel was attempted but failed.

### Reason:

Vercel uses a **serverless architecture** with execution time limits.

The `/v1/api/save-url` route performs:

- URL scraping
- Content parsing
- Chunking
- Embedding generation
- Vector storage in Qdrant

These operations are **long-running and CPU/network intensive**, which exceed Vercel’s serverless timeout limits.

As a result:

- Requests timeout
- Functions terminate early
- Embeddings are not fully stored

This architecture is not ideal for heavy ingestion pipelines.

---

## Render Deployment (Working with Limitations)

The backend is successfully deployed on Render:

 https://linkmind.onrender.com

However:

- The free tier has limited CPU resources.
- Long-running ingestion (`/v1/api/save-url`) may still timeout.
- Scaling is not available on the free tier.

Because of architectural limitations (small chunk size, embedding generation, multiple vector writes), ingestion may fail under constrained resources.

To fully support ingestion reliably, the project would require:

- Higher compute plan
- Increased timeout limits

---

# Architectural Limitations

## 1. LLM Support Limitation

Currently:

- Only `backend: "custom"` is supported.
- OpenAI integration is not enabled because no API key was available during development/testing.
- Claude (Anthropic) integration is also not implemented for the same reason.

This means:

- You must provide your own compatible LLM endpoint.
- The backend assumes an OpenAI-style `/chat/completions` interface.

Future versions may include:

- Native OpenAI integration
- Native Claude integration
- OpenRouter support

---

## 2. Custom LLM Responsibility

Since this project relies on a user-provided LLM endpoint:

- You are responsible for hosting and maintaining the LLM server.
- You must ensure it follows the required request/response format.
- Authentication must be handled via the `api_key` header.

---

# How to Get a Free LLM Endpoint (For Testing)

If you don’t have your own hosted LLM server, you can use **OpenRouter** to test this project for free.

### Steps:

1. Create a free account at:
   https://openrouter.ai

2. Go to the API Keys section and generate a new API key.

3. Browse available models and search for:
   - Free models
   - Models labeled as `:free`

4. Use the OpenRouter Chat Completions endpoint:

```
https://openrouter.ai/api/v1/chat/completions
```

5. When calling `/v1/api/query`, set:

```json
{
  "query": "Your question",
  "backend": "custom",
  "model": "openrouter-model-id",
  "custom_url": "https://openrouter.ai/api/v1/chat/completions"
}
```

6. Add your OpenRouter API key in the request header:

```
api_key: YOUR_OPENROUTER_API_KEY
```

---

>   Note:
> - Make sure the selected model supports chat completions.
> - Free models may have rate limits.
> - Model availability may change over time.


## 3. Retrieval Limitations

- Chunk size is currently small (200 characters).
- No chunk overlap is used.
- Retrieval limit is capped at 20 vectors.
- Context is truncated to 4000 characters before being sent to the LLM.

This may:

- Reduce answer completeness
- Affect long-article understanding
- Limit deep contextual reasoning

These values can be tuned for better performance.

---

## 4. Data Management Limitation

- UUIDs are used for vector IDs.
- Repeated ingestion of the same URLs will create duplicate embeddings.
- There is currently no deduplication or cleanup mechanism.

For production use, you may want:

- URL hashing
- Duplicate detection
- Collection reset endpoint
- Metadata-based filtering

---

## Environment Variables

- `QdrantClientURL` — Qdrant Cloud URL  
- `QdrantClientAPIKey` — Qdrant API key  

---

## Sources

- [YouTube](https://www.youtube.com/watch?v=d4yCWBGFCEs&list=LL&index=9)
- [Virtual environments | FastAPI](https://fastapi.tiangolo.com/virtual-environments/)
- [FastAPI Async API Guide](https://www.mindbowser.com/fastapi-async-api-guide/)
- [What is OpenRouter](https://www.codecademy.com/article/what-is-openrouter/)
- [Deploy FastAPI on Vercel](https://dev.to/highflyer910/deploy-your-fastapi-app-on-vercel-the-complete-guide-27c0)
- [FastAPI Testing Events](https://fastapi.tiangolo.com/advanced/testing-events/)
- [Qdrant Cloud Quickstart](https://qdrant.tech/documentation/cloud-quickstart/)
- [Related deployment issue I went through](https://community.vercel.com/t/cannot-install-needed-python-library-requirements/732)
- [Deploy to Render for free](https://render.com/docs/free)


---

**Architecture Notes**
1. RAG (Retrieval-Augmented Generation) architecture  
2. Vector-based semantic search with Qdrant  
3. Chunk → Embed → Store embedding pipeline  
4. Stateless FastAPI backend  
5. LLM integration with context augmentation  

---
**Additional Note**

> This is a beginner-level learning project built to understand how RAG (Retrieval-Augmented Generation), FastAPI, vector databases (Qdrant), and custom LLM integrations work together. It is intended for experimentation and getting started with AI-powered backend systems, not for production use.

---

**Apology & feedback**

Sorry for any mistakes, oversights, or rough edges in this project—whether in the code, docs, or setup. If you spot a bug, have a suggestion, or want to report an issue, please feel free to reach out or open an issue; your feedback is welcome and appreciated.
