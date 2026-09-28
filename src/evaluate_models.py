"""Phase 7: score every model the same way.

Usage:
    python -m src.evaluate_models metrics              # token F1, ROUGE-L, BERTScore -> results/metrics.csv
    python -m src.evaluate_models judge [--limit N]    # LLM judge -> results/judge.csv (resumes)
    python -m src.evaluate_models handsheet            # 100 blinded rows to grade by hand
    python -m src.evaluate_models agreement            # you vs. the judge (accuracy, Cohen's kappa)
    python -m src.evaluate_models report               # model x task tables with bootstrap 95% CIs
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
RESULTS = Path("results")

# display name -> (results file, strategy filter or None)
SYSTEMS = {
    "gpt-4.1-mini zero-shot": ("baseline.csv", "zero_shot"),
    "gpt-4.1-mini few-shot": ("baseline.csv", "few_shot"),
    "gpt-4.1-mini + retrieval": ("retrieval.csv", None),
    "Qwen 1.5B base (Colab 4-bit)": ("base_small.csv", None),
    "Qwen 1.5B fine-tuned (Colab 4-bit)": ("finetuned.csv", None),
    "Qwen 1.5B base, Q4 GGUF": ("gguf_base-Q4_K_M.csv", None),
    "Qwen 1.5B fine-tuned, Q8 GGUF": ("gguf_model-Q8_0.csv", None),
    "Qwen 1.5B fine-tuned, Q4 GGUF": ("gguf_model-Q4_K_M.csv", None),
}
EXTRACTION = {"Named Entity Recognition", "Abbreviation Expansion", "Relation Extraction",
              "Temporal Information Extraction", "Coreference Resolution"}
GENERATION = {"Summarization", "Paraphrasing"}

JUDGE_MODEL = "gpt-4.1-2025-04-14"  # stronger than the gpt-4.1-mini contestant
JUDGE_PRICES = (2.00, 8.00)          # USD per 1M tokens (input, output); check openai.com/api/pricing
JUDGE_WORKERS = 3                    # the account allows 30k tokens/min on gpt-4.1

JUDGE_PROMPT = """You are grading an answer to a question about a clinical note.

Note:
{note}

Question: {question}
Reference answer: {reference}
Candidate answer: {prediction}

Score the candidate:
- correctness: 2 = fully correct, 1 = partly correct, 0 = wrong
- faithful: 1 if every claim is supported by the note, 0 if anything is made up
Return only JSON: {{"correctness": 0-2, "faithful": 0 or 1, "reason": "one sentence"}}"""


def load_all():
    """Every system's 400 predictions stacked into one table, `model` column first."""
    frames = []
    for name, (file, strategy) in SYSTEMS.items():
        df = pd.read_csv(RESULTS / file)
        if strategy:
            df = df[df.strategy == strategy]
        assert len(df) == 400 and df.row_id.is_unique, f"{name}: {len(df)} rows"
        frames.append(df.assign(model=name))
    cols = ["model", "row_id", "task", "question", "answer", "prediction",
            "input_tokens", "output_tokens", "latency_s"]
    out = pd.concat(frames, ignore_index=True)[cols]
    out["prediction"] = out.prediction.fillna("")
    return out


# ---------- 1. automatic metrics ----------

def _tokens(s):
    return re.findall(r"\w+", str(s).lower())


def token_f1(pred, ref):
    p, r = _tokens(pred), _tokens(ref)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def metrics():
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    df = load_all()
    df["token_f1"] = [token_f1(p, a) for p, a in zip(df.prediction, df.answer)]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    df["rougeL"] = [rouge.score(a, p)["rougeL"].fmeasure for p, a in zip(df.prediction, df.answer)]

    # BERTScore only where wording is expected to differ from the reference
    gen = df.task.isin(GENERATION)
    _, _, f1 = bert_score(df.loc[gen, "prediction"].tolist(), df.loc[gen, "answer"].tolist(),
                          lang="en", rescale_with_baseline=True, batch_size=32, verbose=True)
    df.loc[gen, "bertscore"] = f1.numpy()
    df.to_csv(RESULTS / "metrics.csv", index=False)
    print(df.groupby("model", sort=False)[["token_f1", "rougeL", "bertscore"]].mean().round(3).to_string())


# ---------- 2. LLM judge ----------

def _judge_one(client, note, r):
    prompt = JUDGE_PROMPT.format(note=note, question=r.question, reference=r.answer,
                                 prediction=r.prediction)
    usage = [0, 0]
    for attempt in range(2):  # retry once if the reply isn't valid JSON
        resp = client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0, max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        usage[0] += resp.usage.prompt_tokens
        usage[1] += resp.usage.completion_tokens
        try:
            out = json.loads(resp.choices[0].message.content)
            correctness, faithful = int(out["correctness"]), int(out["faithful"])
            assert correctness in (0, 1, 2) and faithful in (0, 1)
            return {"correctness": correctness, "faithful": faithful, "reason": out.get("reason", ""),
                    "judge_input_tokens": usage[0], "judge_output_tokens": usage[1]}
        except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
            continue
    return {"correctness": np.nan, "faithful": np.nan, "reason": "unparseable judge reply",
            "judge_input_tokens": usage[0], "judge_output_tokens": usage[1]}


def judge(limit=None, out=RESULTS / "judge.csv"):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(max_retries=10)  # backs off and retries on rate limits (429)
    notes = pd.read_parquet(DATA / "eval_400.parquet").note
    df = load_all()
    if limit:  # the same first N questions for every model
        df = df[df.row_id.isin(df.row_id.unique()[:limit])]
    if out.exists():
        done = set(zip(*pd.read_csv(out)[["model", "row_id"]].T.values))
        df = df[[(m, r) not in done for m, r in zip(df.model, df.row_id)]]
    print(f"judging {len(df)} answers with {JUDGE_MODEL}")

    buffer = []
    def flush():
        if buffer:
            pd.DataFrame(buffer).to_csv(out, mode="a", header=not out.exists(), index=False)
            buffer.clear()

    with ThreadPoolExecutor(JUDGE_WORKERS) as pool:
        futures = {pool.submit(_judge_one, client, notes.loc[r.row_id], r): r for r in df.itertuples()}
        for i, f in enumerate(as_completed(futures), 1):
            r = futures[f]
            try:
                buffer.append({"model": r.model, "row_id": r.row_id, "task": r.task, **f.result()})
            except Exception as e:  # API error: skip, a rerun retries it
                print(f"{r.model} / {r.row_id} failed: {e}")
            if len(buffer) >= 50:
                flush()
                print(f"{i}/{len(df)} judged", flush=True)
    flush()

    res = pd.read_csv(out)
    cost = (res.judge_input_tokens.sum() * JUDGE_PRICES[0] + res.judge_output_tokens.sum() * JUDGE_PRICES[1]) / 1e6
    print(f"{len(res)} judged so far, {res.correctness.isna().sum()} unparseable, judge cost so far ${cost:.2f}")
    print(res.groupby("model", sort=False)[["correctness", "faithful"]].mean().round(3).to_string())


# ---------- 3. hand grading ----------

HAND = RESULTS / "hand_grade.csv"
HAND_KEY = RESULTS / "hand_grade_key.csv"


def handsheet(n=100):
    """100 rows across models and tasks, shuffled, model names hidden (kept in a separate key file)."""
    if HAND.exists():
        raise SystemExit(f"{HAND} already exists; delete it first if you really want a new sample")
    notes = pd.read_parquet(DATA / "eval_400.parquet").note
    df = load_all()
    # every model gets one row per task, then the rest are filled at random
    sample = df.groupby(["model", "task"]).sample(1, random_state=0)
    extra = df.drop(sample.index).sample(n - len(sample), random_state=0)
    sample = pd.concat([sample, extra]).sample(frac=1, random_state=1).reset_index(drop=True)
    sample.insert(0, "item", range(1, len(sample) + 1))

    sheet = sample[["item", "task", "question", "answer", "prediction"]].rename(
        columns={"answer": "reference_answer", "prediction": "candidate_answer"})
    sheet.insert(2, "note", sample.row_id.map(notes).values)
    sheet["correctness_0_2"] = ""
    sheet["faithful_0_1"] = ""
    sheet["reference_wrong_y_n"] = ""
    sheet.to_csv(HAND, index=False)
    sample[["item", "model", "row_id"]].to_csv(HAND_KEY, index=False)
    print(f"wrote {HAND} ({len(sheet)} rows) and {HAND_KEY} (don't open the key until you've graded)")
    print(sample.model.value_counts().to_string())


def agreement():
    from sklearn.metrics import cohen_kappa_score

    human = pd.read_csv(HAND).merge(pd.read_csv(HAND_KEY), on="item")
    human = human.dropna(subset=["correctness_0_2", "faithful_0_1"])
    j = pd.read_csv(RESULTS / "judge.csv")
    both = human.merge(j, on=["model", "row_id"]).dropna(subset=["correctness"])
    print(f"{len(both)} rows graded by both you and the judge")
    hc, jc = both.correctness_0_2.astype(int), both.correctness.astype(int)
    hf, jf = both.faithful_0_1.astype(int), both.faithful.astype(int)
    print(f"correctness: exact agreement {(hc == jc).mean():.0%}, "
          f"Cohen's kappa {cohen_kappa_score(hc, jc):.2f}, "
          f"weighted kappa {cohen_kappa_score(hc, jc, weights='linear'):.2f}")
    print(f"faithful:    agreement {(hf == jf).mean():.0%}, Cohen's kappa {cohen_kappa_score(hf, jf):.2f}")
    print("\nconfusion (rows = you, columns = judge):")
    print(pd.crosstab(hc, jc).to_string())
    # does the judge favor the OpenAI contestant? compare judge - human by model family
    both["gap"] = jc - hc
    both["family"] = np.where(both.model.str.startswith("gpt"), "gpt-4.1-mini", "Qwen")
    print("\njudge minus you, mean correctness gap by family (positive = judge more generous):")
    print(both.groupby("family").gap.agg(["mean", "count"]).round(2).to_string())
    ref = human.reference_wrong_y_n.astype(str).str.lower().str.startswith("y")
    print(f"\nreference answer marked wrong: {ref.mean():.0%} of {len(human)} rows")


# ---------- 4-5. report ----------

def bootstrap_ci(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def report():
    m = pd.read_csv(RESULTS / "metrics.csv")
    j = pd.read_csv(RESULTS / "judge.csv")
    df = m.merge(j[["model", "row_id", "correctness", "faithful"]], on=["model", "row_id"], how="left")
    df["correct_share"] = df.correctness / 2  # 0-2 scale -> 0-1, reads as "share correct"

    rows = []
    for name, g in df.groupby("model", sort=False):
        rows.append({"model": name, "judged": int(g.correctness.notna().sum()),
                     "judge correctness (0-1)": fmt(*bootstrap_ci(g.correct_share)),
                     "faithful": fmt(*bootstrap_ci(g.faithful)),
                     "token F1": fmt(*bootstrap_ci(g.token_f1))})
    overall = pd.DataFrame(rows)
    print("overall, mean [95% bootstrap CI]:")
    print(overall.to_string(index=False))

    by_task = df.pivot_table(index="task", columns="model", values="correct_share", sort=False).round(2)
    print("\njudge correctness (0-1) by task:")
    print(by_task.T.to_string())
    auto = pd.concat([
        df[df.task.isin(EXTRACTION | {"Question Answering"})].pivot_table(index="model", columns="task", values="token_f1", sort=False),
        df[df.task.isin(GENERATION)].pivot_table(index="model", columns="task", values="rougeL", sort=False).add_suffix(" (ROUGE-L)"),
        df[df.task.isin(GENERATION)].pivot_table(index="model", columns="task", values="bertscore", sort=False).add_suffix(" (BERTScore)"),
    ], axis=1).round(3)
    print("\nautomatic metric by task (token F1 unless noted):")
    print(auto.to_string())

    overall.to_csv(RESULTS / "eval_overall.csv", index=False)
    by_task.to_csv(RESULTS / "eval_by_task_judge.csv")
    auto.to_csv(RESULTS / "eval_by_task_auto.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["metrics", "judge", "handsheet", "agreement", "report"])
    ap.add_argument("--limit", type=int, help="judge only the first N questions (per model)")
    args = ap.parse_args()
    if args.step == "metrics":
        metrics()
    elif args.step == "judge":
        judge(args.limit)
    elif args.step == "handsheet":
        handsheet()
    elif args.step == "agreement":
        agreement()
    else:
        report()


if __name__ == "__main__":
    main()
