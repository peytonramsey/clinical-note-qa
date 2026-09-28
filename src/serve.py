"""Phase 9: HTTP API for the fine-tuned model.

Run:  uvicorn src.serve:app --reload    then open http://localhost:8000/docs
"""
import os
import time
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.prompts import SYSTEM
from src.retrieve import bm25_rank, split_sections

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model-Q4_K_M.gguf")
MAX_TOKENS = 512
TOP_K = 3

app = FastAPI(title="Clinical Note QA",
              description="Answers questions about a discharge summary with a fine-tuned Qwen2.5-1.5B "
                          "(Q4 GGUF). Trained on synthetic notes; not for clinical use.")


@lru_cache(maxsize=1)
def get_llm():
    """Load the model on the first request, so importing this module (e.g. in tests) stays cheap."""
    from llama_cpp import Llama
    return Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)


class Query(BaseModel):
    note: str = Field(min_length=1, description="The discharge summary text")
    question: str = Field(min_length=1)
    use_retrieval: bool = Field(False, description=f"Send only the top {TOP_K} BM25-ranked sections")


class Answer(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}


@app.post("/answer", response_model=Answer)
def answer(q: Query):
    start = time.perf_counter()
    if q.use_retrieval:
        sections = split_sections(q.note)
        top = sorted(bm25_rank(sections, q.question)[:TOP_K])  # keep note order
        context = [sections[i] for i in top]
    else:
        context = [q.note]
    out = get_llm().create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": "Note:\n" + "\n\n".join(context) + f"\n\nQuestion: {q.question}"}],
        max_tokens=MAX_TOKENS, temperature=0)
    return Answer(answer=out["choices"][0]["message"]["content"].strip(),
                  sources=context,
                  latency_ms=(time.perf_counter() - start) * 1000)
