from fastapi import FastAPI, Request
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

from langchain_community.document_loaders import UnstructuredURLLoader
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import HTTPException
from typing import List

import os
from dotenv import load_dotenv

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
    if(collection.name != "News"):
        client.create_collection(
            collection_name="News",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

model = TextEmbedding('BAAI/bge-small-en-v1.5')

class URL(BaseModel):
    urls:List[str]

# print("Checking if command is okay:::RUNNING==>")
base_url = "/v1/api"
@app.get(f"{base_url}/query")
def get_result(request: Request):
    if request.query_params:
        query_text=request.query_params.get("value")
        query_vector = next(iter(model.embed(query_text)))
        results = client.query_points(
            collection_name="News",
            query=query_vector,
            with_payload=True,
            limit=5
        )
        return {
            "status": "query received",
            "query": request.query_params.get("value"),
            "result":results
        }
    return "No query params, running"

@app.post(f"{base_url}/save-url")
def save_url(data: URL):
    urls=data.urls

    try:
        loader = UnstructuredURLLoader(
        urls=urls,
        mode="single",  # or "elements" for more granular control
        show_progress_bar=True,
        headers={"User-Agent": "Mozilla/5.0"},  # Add headers to avoid blocks
        strategy="fast"  # Use fast parsing strategy
        )
        data=loader.load()

        rec_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ", ".", ""],
        chunk_size=200,
        chunk_overlap=0
        )
        all_chunks = []
        metadata = []  # Store metadata for each chunk

        for i, doc in enumerate(data):
            # Split each document
            chunks = rec_splitter.split_text(doc.page_content)
            all_chunks.extend(chunks)
            
            # Store metadata (source URL for each chunk)
            for chunk in chunks:
                metadata.append({
                    "source_url": urls[i],
                    "doc_index": i,
                    "chunk_size": len(chunk)
                })

        

        # embedding generator
        points = []
    
        embeddings = model.embed(all_chunks)

        points = []
        for i, emb in enumerate(embeddings):
            points.append(
                PointStruct(
                    id=i,
                    vector=emb.tolist(),
                    payload={"text": all_chunks[i]}
                )
            )
        # upsert points to collection
        client.upsert(
        collection_name="News",
        points=points,
        )

        return {"message": "Data stored successfully.", "URL" : "texts"}
    except Exception as e:
        raise HTTPException(
            status_code=400,   # or 500 depending on the case
            detail="Failed to store data"
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
#     url="https://8e00a7c3-4a70-4652-a44d-0308f818bb50.europe-west3-0.gcp.cloud.qdrant.io:6333", 
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.W70mqTWbnVbLuq1XlGFLRhjRL7K_0aDrZ3SqMhELVeo",
# )

# print(qdrant_client.get_collections())