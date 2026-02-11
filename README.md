# News AI Backend

Backend for saving news URLs and querying them with AI-powered search.

## Routes

| Route | Method | Use |
|-------|--------|-----|
| `/v1/api/save-url` | POST | Ingest news URLs; content is scraped, chunked, embedded, and stored in Qdrant. |
| `/v1/api/query` | POST | Search stored news and (with `backend: "custom"`) get an AI answer from your LLM. |

## Payloads

**POST `/v1/api/save-url`**
```json
{
  "urls": ["https://example.com/article1", "https://example.com/article2"]
}
```

**POST `/v1/api/query`**  
Header: `api_key` (for custom LLM).  
Body:
```json
{
  "query": "Your question",
  "backend": "custom",
  "model": "your-model-id",
  "custom_url": "https://your-llm-api/completions"
}
```

## Env

- `QdrantClientURL`, `QdrantClientAPIKey` — Qdrant connection.
