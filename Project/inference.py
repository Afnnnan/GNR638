#!/usr/bin/env python3
"""
GNR638 - Deep Learning MCQ Solver — Inference Script
=====================================================

Competition : GNR638 Project - Visual MCQ Answering
Author      : Afnan Abdul Gafoor (22b2505)

Approach    : Zero-shot VLM inference using Qwen3-VL-8B-Instruct
              4-bit NF4 quantization via BitsAndBytes
              No task-specific training; pure reasoning over MCQ images

Usage:
    python inference.py --test_dir /path/to/test_dir

References:
    - Qwen3-VL Technical Report: https://arxiv.org/abs/2511.21631
    - Qwen3-VL GitHub: https://github.com/QwenLM/Qwen3-VL
    - BitsAndBytes (4-bit quantization): https://github.com/TimDettmers/bitsandbytes
    - HuggingFace Transformers: https://github.com/huggingface/transformers
    - qwen-vl-utils: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils
"""

import argparse
import gc
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — Unified settings (same as Kaggle T4 run)
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Image resolution caps (pixels = width * height)
MIN_PIXELS = 128 * 128        # floor: don't under-sample clean LaTeX images
MAX_PIXELS = 672 * 672        # higher resolution for better OCR (~451K px)

# Generation parameters
TEMPERATURE        = 0.15     # slightly above greedy to avoid repetition loops
MAX_NEW_TOKENS     = 768      # sufficient for full chain-of-thought on DL MCQs
REPETITION_PENALTY = 1.05     # prevents output loops; Qwen docs suggest ~1.05

# Attention backend
ATTN_IMPL = "sdpa"            # PyTorch SDPA — universally supported, faster than eager

# Quantization — always 4-bit NF4
TORCH_DTYPE = torch.float16
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best 4-bit quality
    bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4 GB
    bnb_4bit_compute_dtype=torch.float16, # FP16 compute
)

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = (
    "You are an expert in deep learning with thorough knowledge of "
    "neural network architectures, optimization, backpropagation, "
    "CNNs, RNNs, Transformers, and related topics.\n\n"
    "You will be shown an image containing a multiple-choice question (MCQ).\n\n"
    "Instructions:\n"
    "1. Read the question title, question body, and all four options carefully.\n"
    "2. Transcribe any formulas, code, or mathematical expressions exactly.\n"
    "3. Reason through the problem step by step - show your work.\n"
    "4. At the very end, output ONLY a single digit on its own line:\n"
    "      1  →  option A is correct\n"
    "      2  →  option B is correct\n"
    "      3  →  option C is correct\n"
    "      4  →  option D is correct\n"
    "      5  →  genuinely uncertain (safe skip - no penalty)\n\n"
    "IMPORTANT: Your absolute final line must be exactly one digit (1/2/3/4/5) "
    "and nothing else. Do not write anything after the digit."
)

USER_PROMPT = (
    "Examine this deep learning MCQ image carefully. "
    "Think step by step, then output your final answer as a single digit "
    "(1=A, 2=B, 3=C, 4=D, 5=skip if uncertain)."
)

# =============================================================================
# ANSWER PARSER
# =============================================================================

def parse_answer(text: str) -> int:
    """Extract the MCQ answer digit (1-4) or 5 (skip) from raw model output.
    Never returns a value outside 1-5. Defaults to 5 on any parse failure.
    """
    if not text or not text.strip():
        return 5

    # Step 1: Strip <think>...</think> block (Qwen3 may emit partial thinking markup)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return 5

    # Step 2: Exact match — model output is already a bare digit or letter
    if text in {"1", "2", "3", "4", "5"}:
        return int(text)
    if text.upper() in {"A", "B", "C", "D"}:
        return {"A": 1, "B": 2, "C": 3, "D": 4}[text.upper()]

    letter_to_digit = {"A": 1, "B": 2, "C": 3, "D": 4}

    # Step 3: Numeric answer patterns (in order of specificity)
    num_patterns = [
        r"\bthe\s+(?:correct\s+)?answer\s+is\s+(?:option\s+)?([1-4])\b",
        r"\banswer[:\s]+([1-4])\b",
        r"\boption\s+([1-4])\b",
        r"\b([1-4])\s+(?:is\s+correct|is\s+the\s+(?:correct\s+)?answer)\b",
        r"\bchoose\s+([1-4])\b",
        r"\bselect\s+(?:option\s+)?([1-4])\b",
        r"\*\*([1-4])\*\*",                   # bold markdown
        r"(?:^|\n)\s*([1-4])\s*$",            # digit alone on a line
    ]
    for pat in num_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1))

    # Step 4: Letter answer patterns
    letter_patterns = [
        r"\bthe\s+(?:correct\s+)?answer\s+is\s+(?:option\s+)?([A-D])\b",
        r"\banswer[:\s]+([A-D])\b",
        r"\boption\s+([A-D])\b",
        r"\b([A-D])\s+(?:is\s+correct|is\s+the\s+(?:correct\s+)?answer)\b",
        r"(?:^|\n)\s*([A-D])\s*[.:\)]\s",     # "A. ", "A: ", "A) "
        r"\bchoice\s+([A-D])\b",
        r"\bselect\s+(?:option\s+)?([A-D])\b",
        r"\*\*([A-D])\*\*",                   # bold markdown
        r"(?:^|\n)\s*([A-D])\s*$",            # letter alone on a line
    ]
    for pat in letter_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return letter_to_digit.get(m.group(1).upper(), 5)

    # Step 5: Total parse failure — safe skip (0 points, no penalty)
    return 5


# =============================================================================
# SINGLE-IMAGE INFERENCE
# =============================================================================

def solve_mcq(img_path: Path, model, processor, max_pixels: int = MAX_PIXELS) -> tuple:
    """Run Qwen3-VL on a single MCQ image and return (answer_digit, raw_output).

    Parameters
    ----------
    img_path   : path to the PNG image
    model      : loaded model
    processor  : loaded processor
    max_pixels : upper pixel budget for image patches (reduce to avoid OOM)

    Returns
    -------
    (answer, raw) where answer is in {1,2,3,4,5} and raw is the full model output
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(img_path),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    # Build text input from chat template
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract and preprocess image patches
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize text + encode image patches into model inputs
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=(TEMPERATURE > 0),
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return parse_answer(raw), raw


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNR638 MCQ Solver — Qwen3-VL-8B-Instruct (4-bit NF4)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Absolute path to the test directory containing test.csv and images/",
    )
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────
    test_dir   = Path(args.test_dir)
    images_dir = test_dir / "images"
    test_csv   = test_dir / "test.csv"
    # Output submission.csv in the script's own directory (not test_dir)
    script_dir = Path(__file__).resolve().parent
    output_csv = script_dir / "submission.csv"

    assert test_csv.exists(),   f"test.csv not found at {test_csv}"
    assert images_dir.exists(), f"images/ directory not found at {images_dir}"

    # ── GPU info ───────────────────────────────────────────────────────────
    print(f"PyTorch  : {torch.__version__}")
    print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    print("This will take ~2-3 min (quantizing to 4-bit NF4) ...\n")

    t0 = time.time()

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=TORCH_DTYPE,
        device_map="auto",
        quantization_config=BNB_CONFIG,
        attn_implementation=ATTN_IMPL,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    load_time = time.time() - t0
    print(f"\nModel loaded in {load_time:.1f}s")
    if torch.cuda.is_available():
        used_gb  = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM used after load: {used_gb:.1f} / {total_gb:.1f} GB ({100*used_gb/total_gb:.0f}%)")

    # ── Inference loop ─────────────────────────────────────────────────────
    test_df = pd.read_csv(test_csv)
    total   = len(test_df)
    print(f"\nTest set: {total} images")
    print(test_df.head(), "\n")

    results     = []
    t_run_start = time.time()

    for idx, row in test_df.iterrows():
        image_name = row["image_name"]
        img_num    = idx + 1

        # Try the canonical image path
        img_path = images_dir / f"{image_name}.png"
        if not img_path.exists():
            img_path = test_dir / f"{image_name}.png"  # fallback

        print(f"[{img_num:2d}/{total}] {image_name}", end="  ", flush=True)
        t_img = time.time()

        #  Case 1: Image file not found
        if not img_path.exists():
            print("NOT FOUND → skip (5)")
            results.append({"id": image_name, "image_name": image_name, "option": 5})
            continue

        #  Case 2: Normal inference
        try:
            option, raw = solve_mcq(img_path, model, processor)
            elapsed = time.time() - t_img
            print(f"option={option}  ({elapsed:.1f}s)")
            print(f"    ↳ {raw[:140].strip()!r}")

        #  Case 3: OOM — retry at reduced resolution
        except torch.cuda.OutOfMemoryError:
            print("OOM - retrying at reduced resolution ...")
            success = False

            for scale in [2, 4]:
                gc.collect()
                torch.cuda.empty_cache()
                reduced = MAX_PIXELS // scale
                print(f"    retrying at {int(reduced**0.5)}×{int(reduced**0.5)} (1/{scale} pixels) ...", end="  ")
                try:
                    option, raw = solve_mcq(img_path, model, processor, max_pixels=reduced)
                    elapsed = time.time() - t_img
                    print(f"option={option}  ({elapsed:.1f}s) [reduced res 1/{scale}]")
                    success = True
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM again at 1/{scale}")
                except Exception:
                    print(f"ERROR at 1/{scale}:")
                    traceback.print_exc()
                    option, raw = 5, f"error_oom_retry_1/{scale}"
                    success = True
                    break

            if not success:
                print("All OOM retries exhausted → skip (5)")
                option, raw = 5, "error_oom_final"

        #  Case 4: Unexpected error
        except Exception:
            print("ERROR:")
            traceback.print_exc()
            option, raw = 5, "error_unexpected"

        results.append({"id": image_name, "image_name": image_name, "option": option})

        # Free VRAM between images
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - t_run_start
    print(f"\nInference complete in {total_elapsed/60:.1f} min  ({total_elapsed/total:.1f}s/image avg)")

    # ── Build & save submission ────────────────────────────────────────────
    submission = pd.DataFrame(results)[["id", "image_name", "option"]]

    # Validation
    assert list(submission.columns) == ["id", "image_name", "option"], \
        f"Wrong columns: {list(submission.columns)}"
    assert len(submission) == len(test_df), \
        f"Row count mismatch: {len(submission)} vs {len(test_df)}"
    assert submission["option"].between(1, 5).all(), \
        f"Invalid option values found"

    submission.to_csv(output_csv, index=False)

    # Summary
    dist     = submission["option"].value_counts().sort_index()
    skipped  = int((submission["option"] == 5).sum())
    answered = total - skipped

    print(f"{'='*55}")
    print(f"Saved: {output_csv}  ({len(submission)} rows)")
    print(f"{'='*55}")
    print(submission.to_string(index=False))
    print(f"\nDistribution : { {k: int(v) for k, v in dist.items()} }")
    print(f"Answered     : {answered}/{total}")
    print(f"Skipped      : {skipped}/{total}")
    print(f"\nDone — submission.csv saved to {output_csv}")


if __name__ == "__main__":
    main()
