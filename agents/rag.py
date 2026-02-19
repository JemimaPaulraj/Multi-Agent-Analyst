"""
RAG Agent - Reads PDFs from S3 bucket
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langsmith import traceable
import boto3
from botocore.exceptions import ClientError

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import llm, get_logger
from semantic_cache import check_cache, save_to_cache

logger = get_logger("rag")

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "ticket-forecasting-lake")
S3_PREFIX = os.getenv("S3_PREFIX", "RAG_Data/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Local cache for downloaded PDFs
CACHE_DIR = Path(__file__).resolve().parent.parent / "s3_cache"
VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "vectorstore"

# Session state
_state = {"vectorstore": None, "s3_client": None}


def get_s3_client():
    """Get or create S3 client."""
    if _state["s3_client"] is None:
        _state["s3_client"] = boto3.client("s3", region_name=AWS_REGION)
    return _state["s3_client"]


def download_pdfs_from_s3():
    """Download all PDFs from S3 bucket to local cache."""
    
    logger.info(f"Downloading PDFs from s3://{S3_BUCKET}/{S3_PREFIX}")
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    
    downloaded_files = []
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                
                if not key.lower().endswith('.pdf'):
                    continue
                
                filename = Path(key).name
                local_path = CACHE_DIR / filename
                
                logger.debug(f"Downloading: {key}")
                s3.download_file(S3_BUCKET, key, str(local_path))
                downloaded_files.append(local_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} PDF(s) from S3")
        return downloaded_files
        
    except ClientError as e:
        logger.error(f"S3 Error: {e}")
        return []


def get_vectorstore():
    """Check if vectorstore exists in session or disk."""
    
    if _state["vectorstore"] is not None:
        logger.debug("Using vectorstore from session")
        return _state["vectorstore"]
    
    faiss_index = VECTORSTORE_DIR / "index.faiss"
    if faiss_index.exists():
        logger.info("Loading vectorstore from disk")
        _state["vectorstore"] = FAISS.load_local(
            str(VECTORSTORE_DIR),
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
        return _state["vectorstore"]
    
    return None


def create_vectorstore():
    """Create new vectorstore from PDFs downloaded from S3."""
    
    logger.info("Creating new vectorstore from S3 documents...")
    
    pdf_files = download_pdfs_from_s3()
    
    if not pdf_files:
        logger.warning("No PDFs found in S3")
        return None
    
    # Load all PDFs
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            all_docs.extend(docs)
            logger.debug(f"Loaded {len(docs)} pages from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error loading {pdf_path.name}: {e}")
    
    if not all_docs:
        logger.warning("No documents loaded")
        return None
    
    logger.info(f"Total pages loaded: {len(all_docs)}")
    
    # Step 2/3: Chunk
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3/3: Embed and store
    _state["vectorstore"] = FAISS.from_documents(chunks, OpenAIEmbeddings())
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    _state["vectorstore"].save_local(str(VECTORSTORE_DIR))
    logger.info(f"Saved vectorstore to {VECTORSTORE_DIR}")
    
    return _state["vectorstore"]


@traceable(name="RAG Agent")
def rag_agent(query: str) -> dict:
    """Use vectorstore to answer query with semantic caching."""
    
    try:
        # Step 1: Check semantic cache first
        cached = check_cache(query)
        if cached:
            logger.info(f"CACHE HIT | similarity={cached['similarity']} | similar_query={cached['similar_query'][:50]}...")
            return {
                "agent": "rag_agent",
                "query": query,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "cached": True
            }
        
        # Step 2: Cache miss - do full RAG
        logger.info("CACHE MISS | Running full RAG pipeline")
        
        vectorstore = get_vectorstore()
        if vectorstore is None:
            vectorstore = create_vectorstore()
        
        if vectorstore is None:
            return {
                "agent": "rag_agent",
                "query": query,
                "answer": "No documents loaded. Add PDFs to S3 bucket and restart.",
                "sources": []
            }
        
        # Retrieve with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=4)
        docs = [doc for doc, _ in docs_with_scores]
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_template(
            "Answer based on context. If unsure, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        
        sources = [
            {"source": Path(d.metadata.get("source", "")).name, "page": d.metadata.get("page", "")}
            for d in docs
        ]
        
        # Step 3: Save to cache (includes contexts for RAGAS evaluation)
        contexts = [doc.page_content for doc in docs]
        save_to_cache(query, answer, contexts, sources)
        
        return {
            "agent": "rag_agent",
            "query": query,
            "answer": answer,
            "sources": sources,
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"ERROR: {type(e).__name__}: {e}", exc_info=True)
        return {
            "agent": "rag_agent",
            "query": query,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "error": True
        }
