"""
Semantic Cache using FAISS + DynamoDB

- FAISS: Stores query embeddings for similarity search (finds similar questions)
- DynamoDB: Stores the actual cached answers (retrieves the answer)

Flow:
1. User asks a question
2. Check FAISS for similar questions (semantic search)
3. If similar found → Get answer from DynamoDB (cache hit)
4. If not found → Return None (cache miss)
"""

import os
import hashlib
import time
import json
from pathlib import Path
import boto3
from boto3.dynamodb.types import TypeDeserializer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("CACHE_TABLE", "rag-semantic-cache")
SIMILARITY_THRESHOLD = 0.85  # How similar queries must be (0-1)
CACHE_TTL_HOURS = 24

# Storage paths
CACHE_INDEX_DIR = Path(__file__).resolve().parent.parent / "FAISS_Vectorstore" / "Cache_index"

# In-memory cache for connections 
_cache = {"faiss": None, "dynamodb": None, "embeddings": None}


def check_cache(query: str) -> dict | None:
    """
    Check if a similar question was asked before.
    Returns cached answer if found, None otherwise.
    """
    try:
        # Step 1: Load FAISS index (contains query embeddings)
        if _cache["faiss"] is None:
            index_path = CACHE_INDEX_DIR / "index.faiss"
            if not index_path.exists():
                return None  # No cache exists yet
            _cache["embeddings"] = OpenAIEmbeddings()
            _cache["faiss"] = FAISS.load_local(
                str(CACHE_INDEX_DIR), _cache["embeddings"], allow_dangerous_deserialization=True
            )
        
        # Step 2: Search for similar queries in FAISS
        results = _cache["faiss"].similarity_search_with_score(query, k=1)
        if not results:
            return None
        
        # Step 3: Check if similarity is above threshold
        # FAISS returns L2 distance. For normalized embeddings (OpenAI):
        # - L2 distance 0 = identical (similarity 1.0)
        # - L2 distance 2 = opposite (similarity 0.0)
        # Formula: similarity = 1 - (distance / 2)
        doc, distance = results[0]
        similarity = max(0, 1 - (distance / 2))
        if similarity < SIMILARITY_THRESHOLD:
            return None  # Not similar enough
        
        # Step 4: Get the cached answer from DynamoDB
        if _cache["dynamodb"] is None:
            _cache["dynamodb"] = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)
        
        cache_key = doc.metadata.get("cache_key")
        response = _cache["dynamodb"].get_item(Key={"cache_key": cache_key})
        
        if "Item" not in response:
            return None
        
        # Step 5: Return the cached answer (convert DynamoDB types to JSON-safe)
        item = json.loads(json.dumps(response["Item"], default=str))
        return {
            "answer": item["answer"],
            "sources": item.get("sources", []),
            "cached": True,
            "similar_query": doc.page_content,
            "similarity": round(float(similarity), 3)
        }
        
    except Exception:
        return None  # On any error, treat as cache miss


def save_to_cache(query: str, answer: str, contexts: list = None, sources: list = None):
    """
    Save a new question-answer pair to cache.
    Also stores contexts for offline RAGAS evaluation.
    """
    try:
        # Step 1: Generate cache key from query
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        
        # Step 2: Save answer to DynamoDB
        if _cache["dynamodb"] is None:
            _cache["dynamodb"] = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)
        
        _cache["dynamodb"].put_item(Item={
            "cache_key": cache_key,
            "query": query,
            "answer": answer,
            "contexts": contexts or [],  # For RAGAS evaluation
            "sources": sources or [],
            "created_at": int(time.time()),
            "ttl": int(time.time()) + (CACHE_TTL_HOURS * 3600)  # Auto-expire
        })
        
        # Step 3: Save query embedding to FAISS
        if _cache["embeddings"] is None:
            _cache["embeddings"] = OpenAIEmbeddings()
        
        doc = Document(page_content=query, metadata={"cache_key": cache_key})
        
        if _cache["faiss"] is None:
            # First entry - create new index
            _cache["faiss"] = FAISS.from_documents([doc], _cache["embeddings"])
        else:
            # Add to existing index
            _cache["faiss"].add_documents([doc])
        
        # Step 4: Save FAISS index to disk
        CACHE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _cache["faiss"].save_local(str(CACHE_INDEX_DIR))
        
    except Exception:
        pass  # Cache save failure shouldn't break the app
