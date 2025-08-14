#!/usr/bin/env python3
"""
Batch Dataset Tester for RMLDemo

Runs a small set of dataset-grounded queries against several representative
JSONL files under /Users/elite/R-LLM/data without starting the API server.

Environment is forced to dataset-only (no web search, no world knowledge).
Outputs a compact JSON report per dataset with success flags and snippets.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rml_cli_demo import RMLDemo, Config


DATASETS: List[Path] = [
    Path("/Users/elite/R-LLM/data/test_rml_data.jsonl"),
    Path("/Users/elite/R-LLM/data/validated_rml_data.jsonl"),
    Path("/Users/elite/R-LLM/data/real_training_data.jsonl"),
    Path("/Users/elite/R-LLM/data/real_validation_data.jsonl"),
    Path("/Users/elite/R-LLM/data/streaming/HuggingFaceFW_fineweb_clean.jsonl"),
    Path("/Users/elite/R-LLM/data/world_knowledge/working_wikitext/wikitext_chunk_000002.jsonl"),
]

QUERIES: List[str] = [
    "Give one concise insight based only on the dataset and include Sources.",
    "What topic is discussed in the dataset? Include Sources.",
    "Summarize any topic from the dataset in one sentence and include Sources.",
]


def run_dataset(path: Path, n_entries: int = 50) -> Dict:
    os.environ["RML_DISABLE_WEB_SEARCH"] = "1"
    os.environ["RML_DISABLE_WORLD_KNOWLEDGE"] = "1"
    os.environ["RML_DATASET_PATH"] = str(path)

    config = Config()
    demo = RMLDemo(config)
    try:
        # Cap n_entries to a small number for performance
        demo.load_data(n_entries=n_entries)
    except Exception as e:
        return {
            "dataset": str(path),
            "error": f"load_data failed: {e}",
        }

    results = {"dataset": str(path), "entries": demo.memory.stats.get("total_entries", 0), "queries": []}
    for q in QUERIES:
        try:
            ans = demo.query(q)
            results["queries"].append({
                "q": q,
                "ok": bool(ans and isinstance(ans, str) and len(ans.strip()) > 0),
                "has_sources": ("Sources" in ans) or ("Source" in ans),
                "answer_preview": ans[:300],
            })
        except Exception as e:
            results["queries"].append({
                "q": q,
                "ok": False,
                "error": str(e),
            })
    return results


def main():
    report: List[Dict] = []
    for ds in DATASETS:
        if not ds.exists():
            report.append({"dataset": str(ds), "error": "missing"})
            continue
        report.append(run_dataset(ds))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

