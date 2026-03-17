"""
RAGAS Evaluation (data from DynamoDB).
Run from project root: python rag/eval/ragas_eval.py
"""

import os
import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import math
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import boto3
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("CACHE_TABLE", "rag-semantic-cache")


def get_rag_data_from_dynamodb():
    """Read query, contexts, answer for each item in the cache table."""
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = dynamodb.Table(TABLE_NAME)
    questions, answers, contexts_list = [], [], []

    def add_item(item):
        query = item.get("query") or ""
        answer = item.get("answer") or ""
        ctx = item.get("contexts") or []
        contexts_list.append([str(c) for c in ctx] if isinstance(ctx, (list, tuple)) else [str(ctx)])
        questions.append(query)
        answers.append(answer)

    scan = table.scan()
    for item in scan.get("Items", []):
        add_item(item)
    while "LastEvaluatedKey" in scan:
        scan = table.scan(ExclusiveStartKey=scan["LastEvaluatedKey"])
        for item in scan.get("Items", []):
            add_item(item)

    return questions, answers, contexts_list


def main():
    questions, answers, contexts_list = get_rag_data_from_dynamodb()
    n = len(questions)
    output_file = Path(__file__).resolve().parent / "output.json"

    if n == 0:
        output = {"timestamp": datetime.now().isoformat(), "source": TABLE_NAME, "num_samples": 0, "scores": {}}
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        return

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    })
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        embeddings=OpenAIEmbeddings(),
        show_progress=False,
    )

    raw = getattr(results, "scores", None)
    if raw is None:
        scores = {}
    else:
        if hasattr(raw, "to_list"):
            raw = raw.to_list()
        if not raw:
            scores = {}
        else:
            keys = [k for k in raw[0] if isinstance(raw[0].get(k), (int, float))]
            scores = {}
            for k in keys:
                vals = [s.get(k) for s in raw if isinstance(s.get(k), (int, float)) and not math.isnan(s.get(k))]
                if vals:
                    scores[k] = round(sum(vals) / len(vals), 3)
                else:
                    scores[k] = None
    if scores.get("faithfulness") is not None and not (isinstance(scores["faithfulness"], float) and math.isnan(scores["faithfulness"])):
        scores["hallucination_rate"] = round(1 - scores["faithfulness"], 3)
    else:
        scores["hallucination_rate"] = None

    output = {
        "timestamp": datetime.now().isoformat(),
        "source": TABLE_NAME,
        "num_samples": n,
        "scores": {k: (v if v is not None and not (isinstance(v, float) and math.isnan(v)) else None) for k, v in scores.items()},
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
