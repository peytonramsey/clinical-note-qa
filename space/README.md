---
title: Clinical Note QA
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Fine-tuned 1.5B model answering questions about notes
---

# Clinical Note QA

Ask a question about a discharge summary. The model is
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) fine-tuned with QLoRA
([adapter](https://huggingface.co/peytonramsey/qwen-clinical-note-qa-lora)), merged and quantized to
4-bit GGUF ([weights](https://huggingface.co/peytonramsey/qwen-clinical-note-qa-gguf)), and served on a
free CPU with llama.cpp.

**Not for clinical use.** The model was trained and evaluated only on synthetic notes. It can give wrong
answers and invent details that are not in the note.

## Data and license

Training data and the example notes come from
[Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes)
(Kweon et al., 2023, "Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes"),
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This demo and the
fine-tuned model are shared under the same license, for non-commercial use.
