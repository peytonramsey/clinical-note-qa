"""Phase 6, step 4: time the quantized GGUF models on the 400 evaluation questions (CPU only).

Usage:
    python -m src.benchmark                         # all GGUF models, 400 questions each
    python -m src.benchmark --models model-Q4_K_M --limit 5
    python -m src.benchmark --table                 # just print the Phase 6 results table
Reruns resume: rows already in results/gguf_<model>.csv are skipped.
"""
import argparse
import os
import platform
import time
from pathlib import Path

import pandas as pd

from src.baseline import PRICES, MODEL as API_MODEL
from src.prompts import SYSTEM

DATA = Path("data")
MODELS = Path("models")
RESULTS = Path("results")
GGUFS = ["model-Q8_0", "model-Q4_K_M", "base-Q4_K_M"]
MAX_TOKENS = 512  # same cap as the Phase 5 predictions
SAVE_EVERY = 25


def load(name):
    from llama_cpp import Llama
    return Llama(model_path=str(MODELS / f"{name}.gguf"), n_ctx=4096, n_threads=os.cpu_count(),
                 seed=0, verbose=False)


def ask_local(llm, note, question):
    start = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Note:\n{note}\n\nQuestion: {question}"}],
        max_tokens=MAX_TOKENS, temperature=0)
    return {"prediction": out["choices"][0]["message"]["content"].strip(),
            "input_tokens": out["usage"]["prompt_tokens"],
            "output_tokens": out["usage"]["completion_tokens"],
            "latency_s": time.perf_counter() - start}


def run(name, eval_set):
    out = RESULTS / f"gguf_{name}.csv"
    done = set(pd.read_csv(out).row_id) if out.exists() else set()
    todo = eval_set[~eval_set.index.isin(done)]
    if todo.empty:
        return
    llm = load(name)
    buffer = []
    for row_id, r in todo.iterrows():
        buffer.append({"row_id": row_id, "task": r.task, "question": r.question, "answer": r.answer,
                       "strategy": name, "model": f"{name}.gguf", **ask_local(llm, r.note, r.question)})
        if len(buffer) >= SAVE_EVERY:
            pd.DataFrame(buffer).to_csv(out, mode="a", header=not out.exists(), index=False)
            buffer.clear()
            print(f"{name}: saved through row {row_id}", flush=True)
    if buffer:
        pd.DataFrame(buffer).to_csv(out, mode="a", header=not out.exists(), index=False)


def table():
    """The Phase 6 results table: every column except accuracy (that's Phase 7)."""
    rows = []
    api = pd.read_csv(RESULTS / "baseline.csv")
    api = pd.concat([api[api.strategy == "zero_shot"].assign(label="gpt-4.1-mini, zero-shot"),
                     api[api.strategy == "few_shot"].assign(label="gpt-4.1-mini, few-shot"),
                     pd.read_csv(RESULTS / "retrieval.csv").assign(label="gpt-4.1-mini + retrieval")])
    p_in, p_out = PRICES[API_MODEL]
    for label, g in api.groupby("label", sort=False):
        cost = (g.input_tokens.mean() * p_in + g.output_tokens.mean() * p_out) / 1e6 * 1000
        rows.append({"model": label, "size_on_disk": "API", "median_latency_s": g.latency_s.median(),
                     "tokens_per_s": (g.output_tokens / g.latency_s).median(),
                     "cost_per_1000_q": f"${cost:.2f}"})
    labels = {"base-Q4_K_M": "Base 1.5B (untuned), Q4", "model-Q8_0": "Fine-tuned, Q8",
              "model-Q4_K_M": "Fine-tuned, Q4"}
    for name, label in labels.items():
        path = RESULTS / f"gguf_{name}.csv"
        if not path.exists():
            continue
        g = pd.read_csv(path)
        rows.append({"model": f"{label} (n={len(g)})",
                     "size_on_disk": f"{(MODELS / f'{name}.gguf').stat().st_size / 1e9:.2f} GB",
                     "median_latency_s": g.latency_s.median(),
                     "tokens_per_s": (g.output_tokens / g.latency_s).median(),
                     "cost_per_1000_q": "local"})
    t = pd.DataFrame(rows).round(2)
    print(t.to_string(index=False))
    print(f"\nlocal hardware: {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} threads, CPU only (llama.cpp)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=GGUFS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if not args.table:
        eval_set = pd.read_parquet(DATA / "eval_400.parquet")
        if args.limit:
            eval_set = eval_set.head(args.limit)
        for name in args.models:
            run(name, eval_set)
    table()


if __name__ == "__main__":
    main()
