# Clinical Note QA: Can a Free, Local LLM Replace a Paid API?

Answering questions about hospital discharge summaries with three approaches, compared on **accuracy, hallucinations, speed and cost**:

1. **Frontier API:** OpenAI gpt-4.1-mini, zero-shot and few-shot
2. **API + retrieval (RAG):** the same model sees only the 3 most relevant note sections
3. **Small local model:** Qwen2.5-1.5B-Instruct fine-tuned with QLoRA, quantized to 8-bit and 4-bit GGUF, running on a laptop CPU

**Links:** [fine-tuned adapter](https://huggingface.co/peytonramsey/qwen-clinical-note-qa-lora) · [quantized GGUF models](https://huggingface.co/peytonramsey/qwen-clinical-note-qa-gguf) · [live demo](https://huggingface.co/spaces/peytonramsey/clinical-note-qa)

> Synthetic data only. Not validated on real patient records and not for clinical use.

## Results

400 held-out questions (50 per task × 8 clinical NLP tasks). Accuracy and hallucinations are scored by an LLM judge (gpt-4.1); 95% bootstrap confidence intervals are in brackets.

| System | Judged accuracy | Hallucination rate | Token F1 | Size | Median latency | Cost / 1K questions |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-4.1-mini, zero-shot | **0.916** [0.894, 0.937] | **3.4%** | 0.537 | API | 0.84 s | $0.32 |
| gpt-4.1-mini, few-shot | 0.898 [0.878, 0.918] | 3.1% | 0.578 | API | 0.92 s | $0.68 |
| gpt-4.1-mini + retrieval | 0.855 [0.830, 0.878] | 3.9% | 0.512 | API | 0.82 s | $0.24 |
| Qwen 1.5B fine-tuned, Q8 | 0.711 [0.683, 0.741] | 14.8% | **0.622** | 1.65 GB | 6.44 s | $0 (local CPU) |
| Qwen 1.5B fine-tuned, Q4 | 0.680 [0.642, 0.713]† | 16.2% | 0.614 | 0.99 GB | 5.10 s | $0 (local CPU) |
| Qwen 1.5B base (untuned), Q4 | 0.689 [0.659, 0.719] | 15.2% | 0.499 | 0.99 GB | 4.75 s | $0 (local CPU) |

Judged accuracy counts a partly correct answer as half. A hallucination is an answer containing any claim not supported by the note. Local latency was measured on an Intel Core Ultra 9 288V laptop, CPU only. API costs are estimates at $0.40 / $1.60 per 1M input/output tokens.
†The Q4 row is missing its Temporal Information Extraction questions (the judge's API budget ran out); on the 273 questions judged for every system, the ranking is unchanged.

### Findings

- **Word-overlap metrics picked the wrong winner.** Token F1, ROUGE-L and BERTScore all ranked the fine-tuned 1.5B model first. The LLM judge put gpt-4.1-mini about 20 points ahead in accuracy, with a quarter of the hallucinations. Fine-tuning mostly taught the small model to *sound like* the reference answers.
- **Recommendation:** use the API for this task. The local model is free per question and keeps notes on the device, but at 71% judged accuracy and a 15% hallucination rate it isn't good enough for clinical text.
- **Fine-tuning helped, but modestly with only 2,000 training rows.** Judged accuracy rose from 0.625 to 0.694 with 4-bit (NF4) inference; for the GGUF models (0.689 → 0.711) the gain is within the confidence intervals.
- **Quantization format matters.** The same fine-tuned weights scored higher as a merged Q8/Q4 GGUF than with bitsandbytes NF4 inference: 85% vs. 72% hallucination-free.
- **Retrieval saved 31% of input tokens** but cost 6 accuracy points, mostly on entity extraction and summarization, where the answer spans many sections. BM25 alone found the right section more often than dense or hybrid retrieval (recall@3 0.84 vs. 0.73 / 0.80).
- **Few-shot prompting** raised overlap scores but not judged accuracy, at 2.1× the cost.

Per-task judged accuracy:

| Task | gpt-4.1-mini zero-shot | Qwen fine-tuned Q8 |
| --- | --- | --- |
| Question Answering | 0.98 | 0.88 |
| Summarization | 1.00 | 0.88 |
| Relation Extraction | 0.98 | 0.81 |
| Temporal Information Extraction | 0.93 | 0.74 |
| Paraphrasing | 0.98 | 0.67 |
| Named Entity Recognition | 0.88 | 0.62 |
| Abbreviation Expansion | 0.77 | 0.58 |
| Coreference Resolution | 0.82 | 0.52 |

### Can the judge be trusted?

- **Agreement:** on 92 answers also graded independently by a second, separate LLM grader, the judge agreed on correctness 80% of the time (Cohen's κ = 0.65, substantial agreement) and on hallucinations 96% of the time (κ = 0.86).
- **Bias check:** the judge was no more generous to the OpenAI contestant than to Qwen (mean gap +0.00 vs. +0.02 on a 0–2 scale).
- **Reference quality:** about 12% of audited reference answers in the dataset were themselves wrong, which caps any reference-based metric.

## How it works

| Step | What happens | Code |
| --- | --- | --- |
| Explore | 158,114 QA pairs, 8 tasks, median note 267 words | `eda.ipynb` |
| Split | Train/val/test grouped by note hash (no leakage); a 400-question eval set | `src/split.py` |
| API baseline | gpt-4.1-mini zero-/few-shot, with token counts, latency and cost logged; resumable | `src/baseline.py` |
| Retrieval | Blank-line section splitter; BM25, MiniLM and hybrid (RRF) rankers; recall@k | `src/retrieve.py` |
| Fine-tune | QLoRA on a Colab T4 (r=16, α=32, loss on answer tokens only) | `notebooks/05_finetune.ipynb` |
| Quantize | Merge adapter, convert with llama.cpp, quantize to Q8_0 / Q4_K_M | `src/quantize.py` |
| Benchmark | CPU latency and throughput for each GGUF on the 400 questions | `src/benchmark.py` |
| Evaluate | Token F1, ROUGE-L, BERTScore, LLM judge, κ, bootstrap CIs | `src/evaluate_models.py` |
| Track | One MLflow run per system with params, metrics and predictions | `src/track.py` |
| Serve | FastAPI `POST /answer` with optional retrieval, and `GET /health` | `src/serve.py` |
| Demo | Gradio app on Hugging Face Spaces (free CPU) | `space/` |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env        # plus HF_TOKEN=hf_... to upload models

python -m src.split                        # data splits + eval set
python -m src.baseline                     # API baseline
python -m src.retrieve recall              # retrieval recall@k
python -m src.retrieve run                 # API + retrieval
# fine-tune: run notebooks/05_finetune.ipynb on a Colab T4
python -m src.quantize                     # merge, GGUF, quantize (needs llama.cpp in models/)
python -m src.benchmark                    # local CPU benchmark
python -m src.evaluate_models metrics && python -m src.evaluate_models judge && python -m src.evaluate_models report
python -m src.track && mlflow ui           # experiment tracking
uvicorn src.serve:app                      # API at http://localhost:8000/docs
pytest -q                                  # 16 tests, ~5 s, no model download needed
```

Every long-running step saves as it goes and resumes where it left off.

## Data and license

Data: [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) (Kweon et al., 2023, *Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes*), licensed CC BY-NC-SA 4.0. The fine-tuned model and demo are shared under the same license, for non-commercial use.
