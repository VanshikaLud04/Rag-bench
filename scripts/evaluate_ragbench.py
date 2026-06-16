import asyncio
import json
import logging
import os
from httpx import AsyncClient
from backend.main import app
from backend.services.evaluation.metrics import compute_faithfulness

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def evaluate():
    qa_path = os.path.join(os.path.dirname(__file__), "..", "data", "qa_pairs.json")
    with open(qa_path, "r") as f:
        qa_pairs = json.load(f)

    basic_scores = []
    ragbench_scores = []

    async with AsyncClient(app=app, base_url="http://test") as client:
        for idx, item in enumerate(qa_pairs):
            query = item["query"]
            print(f"Evaluating Query {idx + 1}/{len(qa_pairs)}: {query}")

            try:
                # Basic Pipeline (dense, no retry)
                res_basic = await client.post("/query/", json={"query": query, "strategy": "dense", "top_k": 3})
                data_basic = res_basic.json()
                ans_basic = data_basic.get("answer", "")
                ctx_basic = [c.get("text", "") for c in data_basic.get("retrieved_chunks", [])]
                f_basic = await compute_faithfulness(ans_basic, ctx_basic)
                basic_scores.append(f_basic)

                # RagBench Pipeline (hybrid, with retry)
                res_rb = await client.post("/query/", json={"query": query, "strategy": "hybrid", "top_k": 3})
                data_rb = res_rb.json()
                ans_rb = data_rb.get("answer", "")
                ctx_rb = [c.get("text", "") for c in data_rb.get("retrieved_chunks", [])]
                f_rb = await compute_faithfulness(ans_rb, ctx_rb)
                ragbench_scores.append(f_rb)

                print(f"  Basic Faithfulness: {f_basic:.2f}")
                print(f"  RagBench Faithfulness: {f_rb:.2f} (Retries: {data_rb.get('retries', 0)})")
            except Exception as e:
                logger.error(f"Error evaluating query {idx+1}: {e}")
                continue

    if not basic_scores or not ragbench_scores:
        print("Could not complete evaluation. Ensure background services (Chroma, Ollama) are running.")
        return

    avg_basic = sum(basic_scores) / len(basic_scores) * 100
    avg_rb = sum(ragbench_scores) / len(ragbench_scores) * 100

    print("\n" + "="*50)
    print(f"Basic Pipeline Average Faithfulness: {avg_basic:.1f}%")
    print(f"RagBench Pipeline Average Faithfulness: {avg_rb:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(evaluate())
