"""
RAG Agent - Reads PDFs from S3 bucket
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from botocore.exceptions import ClientError

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import llm

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
    
    print(f"[RAG] Downloading PDFs from s3://{S3_BUCKET}/{S3_PREFIX}")
    
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
                
                print(f"[RAG] Downloading: {key} -> {local_path}")
                s3.download_file(S3_BUCKET, key, str(local_path))
                downloaded_files.append(local_path)
        
        print(f"[RAG] Downloaded {len(downloaded_files)} PDF(s) from S3")
        return downloaded_files
        
    except ClientError as e:
        print(f"[RAG] S3 Error: {e}")
        return []


def get_vectorstore():
    """Check if vectorstore exists in session or disk."""
    
    if _state["vectorstore"] is not None:
        print("[RAG] Using vectorstore from session")
        return _state["vectorstore"]
    
    faiss_index = VECTORSTORE_DIR / "index.faiss"
    if faiss_index.exists():
        print("[RAG] Loading vectorstore from disk")
        _state["vectorstore"] = FAISS.load_local(
            str(VECTORSTORE_DIR),
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
        )
        return _state["vectorstore"]
    
    return None


def create_vectorstore():
    """Create new vectorstore from PDFs downloaded from S3."""
    
    print("[RAG] Creating new vectorstore from S3 documents...")
    
    pdf_files = download_pdfs_from_s3()
    
    if not pdf_files:
        print("[RAG] No PDFs found in S3")
        return None
    
    # Load all PDFs
    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            all_docs.extend(docs)
            print(f"[RAG] Loaded {len(docs)} pages from {pdf_path.name}")
        except Exception as e:
            print(f"[RAG] Error loading {pdf_path.name}: {e}")
    
    if not all_docs:
        print("[RAG] No documents loaded")
        return None
    
    print(f"[RAG] Total pages loaded: {len(all_docs)}")
    
    # Step 2/3: Chunk
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    print(f"[RAG] Created {len(chunks)} chunks")
    
    # Step 3/3: Embed and store
    _state["vectorstore"] = FAISS.from_documents(chunks, OpenAIEmbeddings())
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    _state["vectorstore"].save_local(str(VECTORSTORE_DIR))
    print(f"[RAG] Saved vectorstore to {VECTORSTORE_DIR}")
    
    return _state["vectorstore"]


def rag_agent(query: str) -> dict:
    """Use vectorstore to answer query."""
    
    try:
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
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print(f"[RAG] Context retrieved: {context[:500]}...")
        
        prompt = ChatPromptTemplate.from_template(
            "Answer based on context. If unsure, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        
        sources = [
            {"source": d.metadata.get("source", ""), "page": d.metadata.get("page", "")}
            for d in docs
        ]
        
        return {
            "agent": "rag_agent",
            "query": query,
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"[RAG] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "agent": "rag_agent",
            "query": query,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "error": True
        }
