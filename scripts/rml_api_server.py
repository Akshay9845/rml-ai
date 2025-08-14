#!/usr/bin/env python3
"""
RML Chat API Server
Exposes a minimal GPT-style chat endpoint backed by RMLDemo.

Endpoints:
- GET /health  → basic liveness
- GET /ready   → readiness with loaded entries/device
- POST /chat   → { message: str } → { answer: str, response_ms: int }

Environment variables:
- RML_API_ENTRIES: number of dataset entries to load (default: 200)
"""

import os
import sys
import time
from typing import Optional
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on sys.path so top-level modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from rml_cli_demo import RMLDemo, Config


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    response_ms: int


def create_app() -> FastAPI:
    app = FastAPI(title="RML Chat API", version="1.0.0")

    # Enable permissive CORS for easy local testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global state
    state = {
        "config": None,
        "demo": None,
    }

    @app.on_event("startup")
    def _startup() -> None:
        # Configure for CPU by default; allow overrides
        config = Config()
        # Auto-select device if not overridden: MPS > CUDA > CPU
        auto_device = (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        config.device = os.environ.get("RML_DEVICE", auto_device)

        # Allow overriding dataset path for verification against trained data
        dataset_override = os.environ.get("RML_DATASET_PATH")
        if dataset_override:
            dataset_path = Path(dataset_override)
            if dataset_path.is_dir():
                # Pick the first *.jsonl file if a directory is provided
                jsonl_files = sorted(dataset_path.rglob("*.jsonl"))
                if not jsonl_files:
                    raise FileNotFoundError(f"No .jsonl files found in {dataset_path}")
                config.rml_dataset = jsonl_files[0]
            else:
                config.rml_dataset = dataset_path

        # Allow overriding decoder model (e.g., to use freshly fine-tuned model dir)
        decoder_override = os.environ.get("RML_DECODER_MODEL")
        if decoder_override:
            config.decoder_model = decoder_override

        entries_to_load = int(os.environ.get("RML_API_ENTRIES", "200"))
        # Allow disabling web search for dataset-grounding verification
        if os.environ.get("RML_DISABLE_WEB_SEARCH", "").strip() == "1":
            os.environ["RML_DISABLE_WEB_SEARCH"] = "1"

        demo = RMLDemo(config)
        # This will load entries and precompute embeddings
        demo.load_data(n_entries=entries_to_load)

        state["config"] = config
        state["demo"] = demo

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        demo: Optional[RMLDemo] = state.get("demo")
        config: Optional[Config] = state.get("config")
        entries = 0
        if demo and getattr(demo, "memory", None) and getattr(demo.memory, "stats", None):
            entries = int(demo.memory.stats.get("total_entries", 0))
        return {
            "ready": bool(demo) and entries > 0,
            "entries": entries,
            "device": getattr(config, "device", "unknown"),
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        demo: Optional[RMLDemo] = state.get("demo")
        if demo is None:
            return ChatResponse(answer="Service not ready yet. Please try again shortly.", response_ms=0)

        start_time = time.time()
        answer = demo.query(req.message)
        # Ensure Sources label is consistent
        answer = answer.replace("Source**:", "**Sources**:")
        answer = answer.replace("Sources**:", "**Sources**:")
        # Ensure minimal, one-sentence style, and always include Sources if available
        if answer:
            line = answer.strip().split('\n')[0]
            if not line.endswith(('.', '!', '?')):
                line = line + '.'
            # Preserve sources if present
            if "Sources" in answer or "Source" in answer:
                src = answer[answer.find("Source"):]
                answer = f"{line}\n\n{src}"
            else:
                answer = line
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ChatResponse(answer=answer, response_ms=elapsed_ms)

    return app


app = create_app()


if __name__ == "__main__":
    # For local testing: python scripts/rml_api_server.py
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )

