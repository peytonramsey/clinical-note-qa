"""Phase 3: frontier-model baseline (zero-shot vs few-shot) with cost and latency logging.

Usage:
    python -m src.baseline --limit 20 --strategies zero_shot --out results/trial.csv   # cost check
    python -m src.baseline                                                             # full 400 x 2 run
Reruns resume: rows already in the output CSV are skipped.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.prompts import SYSTEM

load_dotenv()  # reads OPENAI_API_KEY from .env
client = OpenAI()

MODEL = "gpt-4.1-mini-2025-04-14"
# USD per 1M tokens (input, output). Check openai.com/api/pricing before trusting these.
PRICES = {"gpt-4.1-mini-2025-04-14": (0.40, 1.60)}

DATA = Path("data")
RESULTS = Path("results")
SAVE_EVERY = 25



def build_prompt(note, question, examples=None):
    prompt = ""
    for ex in (examples or []):
        prompt += f"Note:\n{ex['note']}\n\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n---\n\n"
    prompt += f"Note:\n{note}\n\nQuestion: {question}\nAnswer:"
    return prompt


def ask(note, question, examples=None, model=MODEL):
    kwargs = {}
    if model.startswith("gpt-4"):  # reasoning models (gpt-5+, o-series) reject temperature
        kwargs["temperature"] = 0
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model, max_completion_tokens=512,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(note, question, examples)}],
        **kwargs)
    return {"prediction": resp.choices[0].message.content,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "latency_s": time.perf_counter() - start}


def cost_usd(df, model=MODEL):
    p_in, p_out = PRICES[model]
    return (df.input_tokens * p_in + df.output_tokens * p_out) / 1e6


def choose_shots(path=RESULTS / "shots.json"):
    """Two short same-task examples from train per task; row ids saved so the run is repeatable."""
    train = pd.read_parquet(DATA / "train.parquet")
    if path.exists():
        ids = json.loads(path.read_text())
        return {t: train.loc[i].to_dict("records") for t, i in ids.items()}
    short = train[train["note"].str.split().str.len() < 300]
    picked = {t: g.sample(2, random_state=0) for t, g in short.groupby("task")}
    path.write_text(json.dumps({t: g.index.tolist() for t, g in picked.items()}, indent=1))
    return {t: g.to_dict("records") for t, g in picked.items()}


def run(eval_set, strategies, out, model=MODEL):
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.row_id, prev.strategy))
    shots = choose_shots() if "few_shot" in strategies else {}

    buffer = []
    def flush():
        if buffer:
            pd.DataFrame(buffer).to_csv(out, mode="a", header=not out.exists(), index=False)
            buffer.clear()

    for strategy in strategies:
        for row_id, r in eval_set.iterrows():
            if (row_id, strategy) in done:
                continue
            examples = shots[r.task] if strategy == "few_shot" else None
            try:
                result = ask(r.note, r.question, examples, model)
            except Exception as e:  # log and keep going; a rerun retries this row
                print(f"row {row_id} {strategy} failed: {e}")
                continue
            buffer.append({"row_id": row_id, "task": r.task, "question": r.question,
                           "answer": r.answer, "strategy": strategy, "model": model, **result})
            if len(buffer) >= SAVE_EVERY:
                flush()
                print(f"saved through row {row_id} ({strategy})")
    flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N eval questions")
    ap.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot"])
    ap.add_argument("--out", type=Path, default=RESULTS / "baseline.csv")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    eval_set = pd.read_parquet(DATA / "eval_400.parquet")
    if args.limit:
        eval_set = eval_set.head(args.limit)
    run(eval_set, args.strategies, args.out, args.model)

    res = pd.read_csv(args.out)
    summary = res.groupby("strategy")[["input_tokens", "output_tokens", "latency_s"]].mean()
    if args.model in PRICES:
        res["cost_usd"] = cost_usd(res, args.model)
        summary["cost_per_q_usd"] = res.groupby("strategy")["cost_usd"].mean()
        summary["cost_per_400_usd"] = summary["cost_per_q_usd"] * 400
    print(len(res), "rows in", args.out)
    print(summary.round(5))


if __name__ == "__main__":
    main()
