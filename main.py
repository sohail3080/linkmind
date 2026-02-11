# ================================================= IMPORT STATEMENTS ================================================================
import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional
import uuid

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastembed import TextEmbedding
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ================================================= CONFIGURATION ====================================================================

load_dotenv()

# Constants
COLLECTION_NAME = "News"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_SIZE = 384
DEFAULT_MAX_CONTEXT_CHARS = 4000
CHUNK_SIZE = 200
CHUNK_OVERLAP = 0
QUERY_LIMIT = 20
MAX_TOKENS = 300


# ================================================= MODELS ===========================================================================


class URL(BaseModel):
    urls: List[str]


class QueryRequest(BaseModel):
    query: str
    # backend: Literal["openai", "openrouter", "custom"]
    backend: Literal["custom"]
    model: str
    custom_url: str | None = None


# ================================================= SERVICES & UTILITIES =============================================================

# connect to qdrant cloud
client = QdrantClient(
    url=os.environ.get("QdrantClientURL"),
    api_key=os.environ.get("QdrantClientAPIKey"),
)

# Embedding Text
model = TextEmbedding(EMBEDDING_MODEL)


# build context from results
def build_context_from_results(
    results, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS
) -> str:
    """
    Flattens retrieved chunks into a single context string.
    Caps size to avoid token overflow.
    """
    texts = []

    for point in results.points:
        payload = point.payload
        if payload and "text" in payload:
            texts.append(payload["text"].strip())

    context = "\n\n".join(texts)

    # optional safety cap
    return context[:max_chars]


# create collection for news
async def createCollection():
    collections = client.get_collections().collections
    collection_names = {c.name for c in collections}

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_SIZE,
                distance=Distance.COSINE,
            ),
        )


# custom chat
async def customChat(
    query: str,
    chunks: list[str] | None = None,
    model: str | None = None,
    apikey: str | None = None,
    custom_url: str | None = None,
):
    context_text = build_context_from_results(chunks)
    system_prompt = f"""INSTRUCTIONS:
    You are a helpful assistant. Answer the question using only the provided context.
    If any relevant or partially related information exists in the context, you MUST mention it, even if it does not fully answer the question.
    Clearly state that the complete answer is not available when the information is incomplete.
    Respond with “I don't know.” ONLY if the context contains no relevant information at all.
    Do not use outside knowledge or make assumptions.
    Keep the response concise and directly related to the question.

    CONTEXT:
    {context_text}
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                custom_url,
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": MAX_TOKENS,
                },
                headers={"Authorization": f"Bearer {apikey}" if apikey else ""},
            )
            response.raise_for_status()
            # return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code, detail="Error fetching data"
        )
    except httpx.RequestError:
        raise HTTPException(status_code=500, detail="Request failed")
    return response.json()


# ================================================= LIFESPAN & FASTAPI ===============================================================


# Startup code
@asynccontextmanager
async def lifespan(app: FastAPI):
    await createCollection()
    yield
    # Shutdown code (optional)
    # client.close() if needed


app = FastAPI(lifespan=lifespan)

# ================================================= ENDPOINTS ========================================================================

# base url
base_url = f"/v1/api"


# query for news
@app.post(f"{base_url}/query")
async def get_result(payload: QueryRequest, request: Request):
    try:
        if payload.query:
            query_text = payload.query
            query_backend = payload.backend
            query_model = payload.model
            query_apikey = request._headers.get("api_key")
            query_custom_url = payload.custom_url

            query_vector = next(iter(model.embed(query_text)))
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                with_payload=True,
                limit=QUERY_LIMIT,
            )

            if not query_backend:
                return {
                    "status": "query received",
                    "query": query_text,
                    "result": results,
                }

            # First lets handle the custom because I currently do not have any working OpenAI/Claude working api key
            if query_backend == "custom":
                answer = await customChat(
                    query=query_text,
                    chunks=results,
                    model=query_model,
                    apikey=query_apikey,
                    custom_url=query_custom_url,
                )
                return {
                    "status": "query received",
                    "query": query_text,
                    "result": answer,
                    "chunks": results,
                }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Failed to fetch data",
        )


# save urls for news
@app.post(f"{base_url}/save-url")
def save_url(data: URL):
    urls = data.urls

    try:
        loader = UnstructuredURLLoader(
            urls=urls,
            mode="single",
            show_progress_bar=True,
            headers={"User-Agent": "Mozilla/5.0"},
            strategy="fast",
        )
        data = loader.load()

        rec_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", ""],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        all_chunks = []
        metadata = []  # Store metadata for each chunk

        for i, doc in enumerate(data):
            # Split each document
            chunks = rec_splitter.split_text(doc.page_content)
            all_chunks.extend(chunks)

            # Store metadata (source URL for each chunk)
            for chunk in chunks:
                metadata.append(
                    {"source_url": urls[i], "doc_index": i, "chunk_size": len(chunk)}
                )

        # embedding
        points = []
        embeddings = model.embed(all_chunks)

        points = []
        for i, emb in enumerate(embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload={
                        "text": all_chunks[i],
                        "source_url": metadata[i]["source_url"],
                        "doc_index": metadata[i]["doc_index"],
                        "chunk_size": metadata[i]["chunk_size"],
                    },
                )
            )
        # upsert points to collection
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        return {"message": "Data stored successfully.", "URL": "texts"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Failed to store data",
        )


# Source articles I read from:(personal note)

# https://www.mindbowser.com/fastapi-async-api-guide/
# https://www.codecademy.com/article/what-is-openrouter/
# https://dev.to/highflyer910/deploy-your-fastapi-app-on-vercel-the-complete-guide-27c0
# https://fastapi.tiangolo.com/advanced/testing-events/
# https://www.youtube.com/watch?v=d4yCWBGFCEs&list=LL&index=9

# Some installed packages info:(personal note)
#  pip install unstructured libmagic python-magic python-magic-bin
