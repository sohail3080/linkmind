from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import HTTPException
from typing import List

import os
from dotenv import load_dotenv

import httpx

# =================================================IMPORT STATEMENTS======================================================================


app = FastAPI()

load_dotenv()

# connect to Qdrant Cloud
client = QdrantClient(
    url=os.environ.get("QdrantClientURL"),
    api_key=os.environ.get("QdrantClientAPIKey"),
)
all_collections = client.get_collections().collections


for collection in all_collections:
    if collection.name != "News":
        client.create_collection(
            collection_name="News",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
def build_context_from_results(results, max_chars: int = 4000) -> str:
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

    # optional safety cap (VERY important in real systems)
    return context[:max_chars]


model = TextEmbedding("BAAI/bge-small-en-v1.5")


class URL(BaseModel):
    urls: List[str]


class QueryRequest(BaseModel):
    query: str
    backend: str
    model: str
    custom_url: str | None = None


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
                    "max_tokens": 300,
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


# print("Checking if command is okay:::RUNNING==>")
base_url = f"/v1/api"


@app.post(f"{base_url}/query")
async def get_result(payload: QueryRequest, request: Request):
    if payload.query:
        query_text = payload.query
        query_backend = payload.backend
        query_model = payload.model
        query_apikey = request._headers.get("api_key")
        query_custom_url = payload.custom_url

        query_vector = next(iter(model.embed(query_text)))
        results = client.query_points(
            collection_name="News", query=query_vector, with_payload=True, limit=20
        )

        if not query_backend:
            return {"status": "query received", "query": query_text, "result": results}

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
                "chunks":results
            }
    return "Please check payload"


@app.post(f"{base_url}/save-url")
def save_url(data: URL):
    urls = data.urls

    try:
        loader = UnstructuredURLLoader(
            urls=urls,
            mode="single",  # or "elements" for more granular control
            show_progress_bar=True,
            headers={"User-Agent": "Mozilla/5.0"},  # Add headers to avoid blocks
            strategy="fast",  # Use fast parsing strategy
        )
        data = loader.load()

        rec_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ".", ""], chunk_size=200, chunk_overlap=0
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

        # embedding generator
        points = []

        embeddings = model.embed(all_chunks)

        points = []
        for i, emb in enumerate(embeddings):
            points.append(
                PointStruct(
                    id=i,
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
            collection_name="News",
            points=points,
        )

        return {"message": "Data stored successfully.", "URL": "texts"}
    except Exception as e:
        raise HTTPException(
            status_code=400,  # or 500 depending on the case
            detail="Failed to store data",
        )


# # pip install unstructured libmagic python-magic python-magic-bin
# # Enable concurrent loading
# loader = UnstructuredURLLoader(
#     urls=urls,
#     mode="single",  # or "elements" for more granular control
#     show_progress_bar=True,
#     headers={"User-Agent": "Mozilla/5.0"},  # Add headers to avoid blocks
#     strategy="fast"  # Use fast parsing strategy
# )
# data=loader.load()


# rec_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n","\n"," ", ".", ""],
#     chunk_size=200,
#     chunk_overlap=0
# )
# all_chunks = []
# metadata = []  # Store metadata for each chunk

# for i, doc in enumerate(data):
#     # Split each document
#     chunks = rec_splitter.split_text(doc.page_content)
#     all_chunks.extend(chunks)

#     # Store metadata (source URL for each chunk)
#     for chunk in chunks:
#         metadata.append({
#             "source_url": urls[i],
#             "doc_index": i,
#             "chunk_size": len(chunk)
#         })


# model = TextEmbedding('BAAI/bge-small-en-v1.5')


# create collection
# if(results):
#     client.create_collection(
#     collection_name="News",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )


# from qdrant_client import QdrantClient

# qdrant_client = QdrantClient(
#     url="",
#     api_key="",
# )

# print(qdrant_client.get_collections())


# Source articles I read from:

# https://www.mindbowser.com/fastapi-async-api-guide/
# https://www.codecademy.com/article/what-is-openrouter/
