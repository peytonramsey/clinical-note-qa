"""Phase 6, steps 1-3: merge the LoRA adapter, convert to GGUF, quantize, and upload to Hugging Face.

Runs on CPU. Needs llama.cpp cloned into models/llama.cpp with llama-quantize built:
    git clone --depth 1 https://github.com/ggml-org/llama.cpp models/llama.cpp
    cmake -S models/llama.cpp -B models/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
    cmake --build models/llama.cpp/build --target llama-quantize -j

Usage:
    python -m src.quantize            # merge, convert, quantize
    python -m src.quantize --upload   # ...and push the GGUF files to Hugging Face
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()  # HF_TOKEN

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "peytonramsey/qwen-clinical-note-qa-lora"
GGUF_REPO = "peytonramsey/qwen-clinical-note-qa-gguf"
MODELS = Path("models")
LLAMA_CPP = MODELS / "llama.cpp"
QUANTS = ["Q8_0", "Q4_K_M"]


def merge(out=MODELS / "merged"):
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, ADAPTER_REPO).merge_and_unload()
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    return out


def download_base(out=MODELS / "base"):
    """The untuned model, converted the same way, for the "Base 1.5B (untuned)" row."""
    AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).save_pretrained(out)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    return out


def to_gguf(hf_dir, name, quants):
    f16 = MODELS / f"{name}-f16.gguf"
    subprocess.run([sys.executable, LLAMA_CPP / "convert_hf_to_gguf.py", hf_dir,
                    "--outfile", f16, "--outtype", "f16"], check=True)
    outs = []
    for q in quants:
        out = MODELS / f"{name}-{q}.gguf"
        subprocess.run([LLAMA_CPP / "build" / "bin" / "llama-quantize", f16, out, q], check=True)
        outs.append(out)
    return outs


def upload(files):
    api = HfApi()
    api.create_repo(GGUF_REPO, exist_ok=True)
    for f in files:
        api.upload_file(path_or_fileobj=f, path_in_repo=f.name, repo_id=GGUF_REPO)
    print(f"https://huggingface.co/{GGUF_REPO}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    files = to_gguf(merge(), "model", QUANTS)
    to_gguf(download_base(), "base", ["Q4_K_M"])  # benchmark only; not uploaded
    for f in sorted(MODELS.glob("*.gguf")):
        print(f"{f.name}: {f.stat().st_size / 1e9:.2f} GB")
    if args.upload:
        upload(files)


if __name__ == "__main__":
    main()
