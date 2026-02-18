"""
Offline RAGAS Evaluation Script
Run: python evaluate_ragas_offline.py
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from agents.rag import get_vectorstore, create_vectorstore, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Test questions - add your own questions here
TEST_QUESTIONS = [
    "What is NET_500?",
    "How do I reset a password?",
    "What are the support ticket categories?",
    "How do I escalate a ticket?",
    "What is the SLA for critical tickets?",
]


def run_rag(query: str) -> dict:
    """Run RAG and return query, answer, and contexts."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        vectorstore = create_vectorstore()
    
    if vectorstore is None:
        return {"query": query, "answer": "No vectorstore", "contexts": []}
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    contexts = [doc.page_content for doc in docs]
    context = "\n\n".join(contexts)
    
    prompt = ChatPromptTemplate.from_template(
        "Answer based on context. If unsure, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    
    return {"query": query, "answer": answer, "contexts": contexts}


def main():
    print("=" * 60)
    print("RAGAS Offline Evaluation")
    print("=" * 60)
    
    # Run RAG for each test question
    print(f"\nRunning {len(TEST_QUESTIONS)} test queries...\n")
    
    questions, answers, contexts = [], [], []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
        result = run_rag(question)
        questions.append(result["query"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        print(f"    Answer: {result['answer'][:100]}...")
    
    # Create dataset for RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })
    
    # Run RAGAS evaluation
    print("\nRunning RAGAS evaluation...")
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    scores = {k: round(v, 3) for k, v in results.items() if isinstance(v, float)}
    for metric, score in scores.items():
        print(f"  {metric}: {score}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(TEST_QUESTIONS),
        "scores": scores,
        "questions": TEST_QUESTIONS,
    }
    
    output_file = f"ragas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
