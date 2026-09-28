"""Phase 4: retrieve the most relevant note sections for each question.

Usage:
    python -m src.retrieve recall   # recall@k table, token savings, 30 gold labels to hand-check
    python -m src.retrieve run      # answer the 400 eval questions from the top-3 hybrid sections
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = Path("data")
RESULTS = Path("results")
MIN_SECTION_WORDS = 8
TOP_K = 3

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def split_sections(note, min_words=MIN_SECTION_WORDS):
    """Split on blank lines, then merge short chunks (lone headers like "Hospital Course:") forward.

    Splitting on every "Header:" line, as the guide suggests, cuts these notes into
    one-line fields ("Age: 73 years"), so blank lines are the better boundary here.
    """
    chunks = [c.strip() for c in re.split(r"\n\s*\n", note) if c.strip()]
    if len(chunks) < 3:
        chunks = [c.strip() for c in note.split("\n") if c.strip()]
    sections, carry = [], ""
    for c in chunks:
        c = f"{carry}\n{c}" if carry else c
        carry = ""
        if len(c.split()) < min_words:
            carry = c
        else:
            sections.append(c)
    if carry:
        if sections:
            sections[-1] += "\n" + carry
        else:
            sections = [carry]
    return sections


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(sections, question):
    bm25 = BM25Okapi([tokens(s) for s in sections])
    return list(np.argsort(-bm25.get_scores(tokens(question)), kind="stable"))


def dense_rank(sections, question):
    emb = embedder().encode(sections + [question], normalize_embeddings=True)
    return list(np.argsort(-(emb[:-1] @ emb[-1]), kind="stable"))


def hybrid_rank(sections, question, k=60):  # reciprocal rank fusion
    scores = {}
    for ranking in (bm25_rank(sections, question), dense_rank(sections, question)):
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0) + 1 / (k + pos + 1)
    return sorted(scores, key=scores.get, reverse=True)


RETRIEVERS = {"bm25": bm25_rank, "dense": dense_rank, "hybrid": hybrid_rank}


def gold_section(sections, answer):
    """Proxy label: the section sharing the most words with the reference answer."""
    ans = set(tokens(answer))
    return int(np.argmax([len(ans & set(tokens(s))) for s in sections]))


def retrieved_note(note, question, k=TOP_K):
    """Top-k hybrid sections, kept in their original note order."""
    sections = split_sections(note)
    top = sorted(hybrid_rank(sections, question)[:k])
    return "\n\n".join(sections[i] for i in top)


def token_f1(pred, ref):
    """Quick overlap score for comparing runs; the real evaluation comes in Phase 7."""
    p, r = tokens(str(pred)), tokens(str(ref))
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def recall_report(eval_set):
    rows = []
    for row_id, r in eval_set.iterrows():
        sections = split_sections(r.note)
        gold = gold_section(sections, r.answer)
        row = {"row_id": row_id, "task": r.task, "n_sections": len(sections), "gold": gold}
        for name, rank in RETRIEVERS.items():
            ranking = rank(sections, r.question)
            for k in (1, 2, 3):
                row[f"{name}@{k}"] = gold in ranking[:k]
        rows.append(row)
    res = pd.DataFrame(rows)

    cols = [f"{n}@{k}" for n in RETRIEVERS for k in (1, 2, 3)]
    print("recall@k, all tasks:")
    print(res[cols].mean().round(3).to_frame("recall").T.to_string())
    print("\nrecall@k by task:")
    print(res.groupby("task")[cols].mean().round(2).to_string())
    print(f"\nsections per note: median {res.n_sections.median():.0f}, "
          f"notes with <= {TOP_K} sections (retrieval keeps everything): {(res.n_sections <= TOP_K).mean():.0%}")

    kept = eval_set.apply(lambda r: len(retrieved_note(r.note, r.question).split()), axis=1)
    full = eval_set.note.str.split().str.len()
    print(f"note words kept by top-{TOP_K} hybrid: {kept.sum() / full.sum():.1%} "
          f"(saves {1 - kept.sum() / full.sum():.1%} of note words)")

    res.to_csv(RESULTS / "retrieval_recall.csv", index=False)

    # 30 proxy labels for you to hand-check: fill in `agree` with y/n
    check = res.sample(30, random_state=0)
    out = []
    for _, c in check.iterrows():
        r = eval_set.loc[c.row_id]
        out.append({"row_id": c.row_id, "task": r.task, "question": r.question, "answer": r.answer,
                    "gold_section": split_sections(r.note)[c.gold], "agree": ""})
    pd.DataFrame(out).to_csv(RESULTS / "gold_check.csv", index=False)
    print(f"wrote {RESULTS / 'gold_check.csv'} for hand-checking")


def run_retrieval(eval_set, out=RESULTS / "retrieval.csv"):
    from src.baseline import run  # imported here so `recall` doesn't need an API key

    eval_set = eval_set.copy()
    eval_set["note"] = eval_set.apply(lambda r: retrieved_note(r.note, r.question), axis=1)
    run(eval_set, ["zero_shot_top3"], out)


def compare(baseline=RESULTS / "baseline.csv", retrieval=RESULTS / "retrieval.csv"):
    b = pd.read_csv(baseline)
    res = pd.concat([b[b.strategy == "zero_shot"], pd.read_csv(retrieval)])
    res["token_f1"] = [token_f1(p, a) for p, a in zip(res.prediction, res.answer)]
    summary = res.groupby("strategy")[["input_tokens", "output_tokens", "latency_s", "token_f1"]].mean()
    print(summary.round(3).to_string())
    full, top = summary.loc["zero_shot", "input_tokens"], summary.loc["zero_shot_top3", "input_tokens"]
    print(f"\ninput tokens saved by retrieval: {1 - top / full:.1%}")
    print("\ntoken F1 by task:")
    print(res.pivot_table(index="task", columns="strategy", values="token_f1").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["recall", "run", "compare"])
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    eval_set = pd.read_parquet(DATA / "eval_400.parquet")
    if args.step == "recall":
        recall_report(eval_set)
    elif args.step == "run":
        run_retrieval(eval_set)
        compare()
    else:
        compare()


if __name__ == "__main__":
    main()
