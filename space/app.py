"""Hugging Face Space: Clinical Note QA demo (fine-tuned Qwen2.5-1.5B, Q4 GGUF, CPU)."""
import json
import os

import gradio as gr
from llama_cpp import Llama

SYSTEM = ("You answer questions about a clinical discharge summary. "
          "Use only information in the note. If the note does not contain "
          "the answer, say so. Be concise.")

llm = Llama.from_pretrained(repo_id="peytonramsey/qwen-clinical-note-qa-gguf",
                            filename="model-Q4_K_M.gguf", n_ctx=4096,
                            n_threads=os.cpu_count(), verbose=False)


def answer(note, question):
    if not note.strip() or not question.strip():
        return "Please paste a discharge summary and ask a question about it."
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Note:\n{note}\n\nQuestion: {question}"}],
        max_tokens=256, temperature=0)
    return out["choices"][0]["message"]["content"].strip()


with open("examples.json") as f:
    examples = json.load(f)

gr.Interface(
    fn=answer,
    inputs=[gr.Textbox(lines=15, label="Discharge summary"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title="Clinical Note QA (synthetic data, not for clinical use)",
    description=("A Qwen2.5-1.5B-Instruct model fine-tuned with QLoRA to answer questions about discharge "
                 "summaries, quantized to 4-bit GGUF and running on a free CPU. Trained and evaluated on "
                 "synthetic notes only; answers can be wrong or made up. Do not use for real patients. "
                 "Answers take roughly 10-30 seconds on the free CPU."),
    examples=examples,
    cache_examples=False,
    flagging_mode="never",
).launch()
