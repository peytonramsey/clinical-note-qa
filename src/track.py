"""Phase 8: log every model/strategy as one MLflow run (params, metrics, prediction file).

Usage:
    python -m src.track        # (re)logs all runs into the "clinical-note-qa" experiment
    mlflow ui                  # then open http://localhost:5000
Rerunning replaces the previous runs, so the experiment always matches the current results.
"""
from pathlib import Path

import mlflow
import pandas as pd

from src.baseline import MODEL as API_MODEL, PRICES
from src.evaluate_models import RESULTS, SYSTEMS, bootstrap_ci

EXPERIMENT = "clinical-note-qa"
MODELS = Path("models")
TRAIN_ROWS = 2000  # rows the current adapter was trained on (bump after the 15k run)
HARDWARE = "Intel Core Ultra 9 288V laptop, 8 threads, CPU only (llama.cpp)"

LORA = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": "peytonramsey/qwen-clinical-note-qa-lora",
        "train_rows": TRAIN_ROWS, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "learning_rate": 2e-4, "epochs": 1, "effective_batch_size": 16}
QWEN_BASE = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "train_rows": 0}

# run name -> params; keys match SYSTEMS in evaluate_models
RUNS = {
    "gpt-4.1-mini zero-shot": ("api-zero-shot", {"model": API_MODEL, "prompt": "zero_shot", "retrieval": "none",
                                                 "quantization": "api", "hardware": "OpenAI API"}),
    "gpt-4.1-mini few-shot": ("api-few-shot", {"model": API_MODEL, "prompt": "few_shot_2_same_task",
                                               "retrieval": "none", "quantization": "api", "hardware": "OpenAI API"}),
    "gpt-4.1-mini + retrieval": ("api-retrieval-top3", {"model": API_MODEL, "prompt": "zero_shot",
                                                        "retrieval": "hybrid_bm25_minilm_top3", "quantization": "api",
                                                        "hardware": "OpenAI API"}),
    "Qwen 1.5B base (Colab 4-bit)": ("qwen-base-colab-nf4", {**QWEN_BASE, "prompt": "zero_shot", "retrieval": "none",
                                                            "quantization": "bnb_nf4", "hardware": "Colab T4 GPU"}),
    "Qwen 1.5B fine-tuned (Colab 4-bit)": ("qwen-lora-colab-nf4", {**LORA, "prompt": "zero_shot", "retrieval": "none",
                                                                  "quantization": "bnb_nf4 + fp32 adapter",
                                                                  "hardware": "Colab T4 GPU"}),
    "Qwen 1.5B base, Q4 GGUF": ("qwen-base-q4", {**QWEN_BASE, "prompt": "zero_shot", "retrieval": "none",
                                                "quantization": "Q4_K_M", "gguf": "base-Q4_K_M.gguf",
                                                "hardware": HARDWARE}),
    "Qwen 1.5B fine-tuned, Q8 GGUF": ("qwen-lora-q8", {**LORA, "prompt": "zero_shot", "retrieval": "none",
                                                      "quantization": "Q8_0", "gguf": "model-Q8_0.gguf",
                                                      "hardware": HARDWARE}),
    "Qwen 1.5B fine-tuned, Q4 GGUF": ("qwen-lora-q4", {**LORA, "prompt": "zero_shot", "retrieval": "none",
                                                      "quantization": "Q4_K_M", "gguf": "model-Q4_K_M.gguf",
                                                      "hardware": HARDWARE}),
}
COMMON = {"eval_set": "data/eval_400.parquet", "eval_questions": 400, "judge": "gpt-4.1-2025-04-14",
          "max_output_tokens": 512, "decoding": "greedy / temperature 0"}


def metrics_for(name, df):
    g = df[df.model == name]
    judged = g.correctness.dropna()
    m = {"judged_answers": len(judged),
         "judge_correctness_0_2": judged.mean(),
         "faithful_rate": g.faithful.mean(),
         "token_f1": g.token_f1.mean(),
         "rougeL": g.rougeL.mean(),
         "bertscore_gen_tasks": g.bertscore.mean(),
         "median_latency_s": g.latency_s.median(),
         "tokens_per_s": (g.output_tokens / g.latency_s).median(),
         "mean_input_tokens": g.input_tokens.mean(),
         "mean_output_tokens": g.output_tokens.mean()}
    mean, lo, hi = bootstrap_ci(g.correctness / 2)
    m.update({"judge_correct_share": mean, "judge_correct_share_ci_low": lo, "judge_correct_share_ci_high": hi})
    if name.startswith("gpt"):
        p_in, p_out = PRICES[API_MODEL]
        m["cost_per_1k_usd"] = (g.input_tokens.mean() * p_in + g.output_tokens.mean() * p_out) / 1e6 * 1000
    else:
        m["cost_per_1k_usd"] = 0.0  # local; hardware is logged as a param
    gguf = RUNS[name][1].get("gguf")
    if gguf:
        m["size_on_disk_gb"] = (MODELS / gguf).stat().st_size / 1e9
    # per-task judge scores, so the model x task table can be rebuilt from MLflow
    for task, t in g.groupby("task"):
        m[f"judge_correct_share/{task.lower().replace(' ', '_')}"] = (t.correctness / 2).mean()
    return {k: float(v) for k, v in m.items() if pd.notna(v)}


def main():
    df = pd.read_csv(RESULTS / "metrics.csv").merge(
        pd.read_csv(RESULTS / "judge.csv")[["model", "row_id", "correctness", "faithful"]],
        on=["model", "row_id"], how="left")

    exp = mlflow.set_experiment(EXPERIMENT)
    for run in mlflow.search_runs([exp.experiment_id], output_format="list"):
        mlflow.delete_run(run.info.run_id)

    for name, (run_name, params) in RUNS.items():
        file, _ = SYSTEMS[name]
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("system", name)
            mlflow.log_params({**COMMON, **params})
            mlflow.log_metrics(metrics_for(name, df))
            mlflow.log_artifact(str(RESULTS / file), artifact_path="predictions")
        print(f"logged {run_name}")

    with mlflow.start_run(run_name="summary-tables"):  # the result tables, for reference
        mlflow.set_tag("system", "summary tables")
        for f in ["eval_overall.csv", "eval_overall_common.csv", "eval_by_task_judge.csv",
                  "eval_by_task_auto.csv", "phase6_table.csv", "hand_grade.csv", "hand_grade_key.csv"]:
            if (RESULTS / f).exists():
                mlflow.log_artifact(str(RESULTS / f))
    print(f"experiment '{EXPERIMENT}' at {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
