"""
Semantic Cache using FAISS + DynamoDB
- FAISS: Stores query embeddings for similarity search
- DynamoDB: Stores the actual cached answers
"""

import os
import hashlib
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("CACHE_TABLE", "rag-semantic-cache")
SIMILARITY_THRESHOLD = 0.92
CACHE_TTL_HOURS = 24

CACHE_INDEX_DIR = Path(__file__).resolve().parent.parent / "cache_index"
_cache_state = {"faiss": None, "dynamodb": None, "embeddings": None}


def _get_dynamodb():
    if _cache_state["dynamodb"] is None:
        _cache_state["dynamodb"] = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)
    return _cache_state["dynamodb"]


def _get_embeddings():
    if _cache_state["embeddings"] is None:
        _cache_state["embeddings"] = OpenAIEmbeddings()
    return _cache_state["embeddings"]


def _get_cache_index():
    """Load or create FAISS cache index."""
    if _cache_state["faiss"] is not None:
        return _cache_state["faiss"]
    
    index_path = CACHE_INDEX_DIR / "index.faiss"
    if index_path.exists():
        _cache_state["faiss"] = FAISS.load_local(
            str(CACHE_INDEX_DIR), _get_embeddings(), allow_dangerous_deserialization=True
        )
    return _cache_state["faiss"]


def _save_cache_index(index):
    """Save FAISS cache index to disk."""
    CACHE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(str(CACHE_INDEX_DIR))
    _cache_state["faiss"] = index


def get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def check_cache(query: str) -> dict | None:
    """
    Check if similar query exists in cache.
    Returns cached answer if found, None otherwise.
    """
    try:
        cache_index = _get_cache_index()
        if cache_index is None:
            return None
        
        # Search for similar queries
        results = cache_index.similarity_search_with_score(query, k=1)
        
        if not results:
            return None
        
        doc, score = results[0]
        similarity = 1 - score  # FAISS returns distance, convert to similarity
        
        # Check if similar enough
        if similarity < SIMILARITY_THRESHOLD:
            return None
        
        # Get answer from DynamoDB
        cache_key = doc.metadata.get("cache_key")
        response = _get_dynamodb().get_item(Key={"cache_key": cache_key})
        
        if "Item" not in response:
            return None
        
        item = response["Item"]
        return {
            "answer": item["answer"],
            "sources": item.get("sources", []),
            "cached": True,
            "similar_query": doc.page_content,
            "similarity": round(similarity, 3)
        }
        
    except Exception:
        return None


def save_to_cache(query: str, answer: str, contexts: list = None, sources: list = None):
    """
    Save query-answer pair to cache.
    
    Also stores retrieved contexts for offline RAGAS evaluation.
    DynamoDB entry can be exported for: question, contexts, answer → RAGAS
    """
    try:
        cache_key = get_cache_key(query)
        
        # Save to DynamoDB (includes contexts for RAGAS evaluation)
        _get_dynamodb().put_item(Item={
            "cache_key": cache_key,
            "query": query,
            "answer": answer,
            "contexts": contexts or [],
            "sources": sources or [],
            "created_at": int(time.time()),
            "ttl": int(time.time()) + (CACHE_TTL_HOURS * 3600)
        })
        
        # Save embedding to FAISS
        cache_index = _get_cache_index()
        from langchain_core.documents import Document
        doc = Document(page_content=query, metadata={"cache_key": cache_key})
        
        if cache_index is None:
            cache_index = FAISS.from_documents([doc], _get_embeddings())
        else:
            cache_index.add_documents([doc])
        
        _save_cache_index(cache_index)
        
    except Exception:
        pass  # Cache save failure shouldn't break the app
