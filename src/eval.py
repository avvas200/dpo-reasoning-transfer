"""GSM8K evaluation utilities for the dpo-reasoning-transfer project.

Design goals:
- **Resumable.** Crashes mid-run don't lose progress. Every problem is
  appended to a `.partial.jsonl` sibling file; a re-run picks up from the
  first unfinished index.
- **Deterministic.** Greedy decoding + fixed subset + a config hash that
  distinguishes cached results from stale ones.
- **Framework-neutral.** Works with an Unsloth `FastLanguageModel` or a
  vanilla HF `AutoModelForCausalLM`. Unsloth's inference-mode call is
  attempted opportunistically and skipped silently otherwise.
- **Import-friendly.** Notebooks call `from src.eval import evaluate_gsm8k`.

Standard usage:

    from src.eval import evaluate_gsm8k
    results = evaluate_gsm8k(
        model, tokenizer,
        n_samples=500,
        output_path="results/baseline_gsm8k_500.json",
        run_metadata={"condition": "baseline"},
    )
    print(f"Accuracy: {results['accuracy']:.1%}")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

GSM8K_PROMPT = (
    "Solve this problem step by step. "
    "End your response with the final numeric answer.\n\n{question}"
)


# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_answer(text: Optional[str]) -> Optional[str]:
    """Extract the final numeric answer from a GSM8K solution or model output.

    Ground-truth format: '<reasoning>\\n#### 42'.
    Model output: fall back to the last numeric token, standard strict-match
    convention. Commas are stripped so '1,234' normalizes to '1234'.
    """
    if text is None:
        return None
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def _normalize(s: Optional[str]) -> Optional[str]:
    """Numeric normalization: '42.0' -> '42', '18.00' -> '18', '3.50' -> '3.5'."""
    if s is None:
        return None
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return ("%f" % f).rstrip("0").rstrip(".")
    except ValueError:
        return s


def answers_match(pred: Optional[str], gt: Optional[str]) -> bool:
    """Strict match with light numeric normalization."""
    if pred is None or gt is None:
        return False
    if pred == gt:
        return True
    return _normalize(pred) == _normalize(gt)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k_subset(n_samples: int = 500, split: str = "test", start: int = 0):
    """Load a fixed, deterministic subset of GSM8K.

    Using the first `n_samples` rows of a split guarantees that paired tests
    (baseline vs. DPO variant) operate on the same problems.
    """
    return load_dataset(
        "openai/gsm8k", "main", split=f"{split}[{start}:{start + n_samples}]"
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _config_hash(config: dict) -> str:
    """Short stable hash for detecting config drift on the same output path."""
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Greedy-decode a single GSM8K response using the model's chat template."""
    messages = [{"role": "user", "content": GSM8K_PROMPT.format(question=question)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def evaluate_gsm8k(
    model,
    tokenizer,
    n_samples: int = 500,
    output_path: Optional[str] = None,
    max_new_tokens: int = 512,
    start: int = 0,
    run_metadata: Optional[dict] = None,
    force_rerun: bool = False,
    checkpoint_every: int = 10,
) -> dict:
    """Evaluate a model on GSM8K with per-problem checkpointing.

    Args:
        model, tokenizer: HF-compatible model and tokenizer.
        n_samples: Number of problems from the split to evaluate on.
        output_path: Destination for the final results JSON. If a matching
            file exists and is complete, we load and return it (unless
            `force_rerun=True`). If a `.partial.jsonl` sibling exists, we
            resume from where it left off.
        max_new_tokens: Generation cap per problem.
        start: Offset into the split (default 0).
        run_metadata: Free-form dict describing this run (base model, DPO
            condition, seed, etc.) — saved into the output for provenance.
        force_rerun: If True, ignore any existing cache and re-evaluate.
        checkpoint_every: Flush the append log to disk every N problems.

    Returns:
        Dict with keys: accuracy, n_samples, n_correct, results (list of
        per-problem dicts), config, run_metadata.
    """
    if run_metadata is None:
        run_metadata = {}

    config = {
        "benchmark": "gsm8k",
        "split": "test",
        "start": start,
        "n_samples": n_samples,
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy",
        "extraction": "last_number_strict",
        "prompt_template_version": 1,
    }
    cfg_hash = _config_hash(config)

    # --- Cache short-circuit ---------------------------------------------
    if output_path and not force_rerun and Path(output_path).exists():
        try:
            with open(output_path) as f:
                cached = json.load(f)
            if (
                cached.get("config", {}).get("_hash") == cfg_hash
                and len(cached.get("results", [])) == n_samples
            ):
                print(f"[eval] Cached results at {output_path} match config; loading.")
                return cached
            print("[eval] Cached file exists but config or size differs; ignoring cache.")
        except json.JSONDecodeError:
            print("[eval] Cached file is not valid JSON; ignoring.")

    # --- Resume state ----------------------------------------------------
    partial_path: Optional[Path] = None
    completed: dict[int, dict] = {}
    if output_path:
        partial_path = Path(output_path).with_suffix(".partial.jsonl")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if partial_path.exists():
            with open(partial_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("_cfg_hash") == cfg_hash and "index" in rec:
                        completed[rec["index"]] = rec
            if completed:
                print(
                    f"[eval] Resuming: {len(completed)}/{n_samples} problems "
                    f"already done in {partial_path}."
                )
            else:
                # Partial file exists but has no matching-config records — start clean.
                partial_path.unlink()

    dataset = load_gsm8k_subset(n_samples=n_samples, start=start)

    # Opportunistic: put Unsloth models into fast-inference mode.
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    # Pre-fill results slots with any resumed records.
    results: list[Optional[dict]] = [None] * n_samples
    for idx, rec in completed.items():
        if 0 <= idx < n_samples:
            results[idx] = rec

    # --- Main loop -------------------------------------------------------
    partial_fh = open(partial_path, "a") if partial_path else None
    try:
        pbar = tqdm(
            range(n_samples),
            desc="Evaluating GSM8K",
            initial=len(completed),
            total=n_samples,
        )
        for i in pbar:
            if results[i] is not None:
                continue
            example = dataset[i]
            question = example["question"]
            gt_answer = extract_answer(example["answer"])

            response = generate_response(
                model, tokenizer, question, max_new_tokens=max_new_tokens
            )
            pred_answer = extract_answer(response)
            is_correct = answers_match(pred_answer, gt_answer)

            rec = {
                "index": i,
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "response": response,
                "correct": is_correct,
                "_cfg_hash": cfg_hash,
            }
            results[i] = rec

            if partial_fh is not None:
                partial_fh.write(json.dumps(rec) + "\n")
                if (i + 1) % checkpoint_every == 0:
                    partial_fh.flush()
                    os.fsync(partial_fh.fileno())

            # Live-updating accuracy on the progress bar
            done = [r for r in results if r is not None]
            if done:
                running_acc = sum(r["correct"] for r in done) / len(done)
                pbar.set_postfix(acc=f"{running_acc:.1%}")
    finally:
        if partial_fh is not None:
            partial_fh.close()

    # --- Aggregate and persist final result ------------------------------
    clean_results = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in results
    ]
    n_correct = sum(1 for r in clean_results if r["correct"])
    output = {
        "accuracy": n_correct / n_samples,
        "n_samples": n_samples,
        "n_correct": n_correct,
        "config": {**config, "_hash": cfg_hash},
        "run_metadata": run_metadata,
        "results": clean_results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        if partial_path is not None and partial_path.exists():
            partial_path.unlink()
        print(
            f"[eval] Saved: {output_path}  "
            f"(accuracy: {output['accuracy']:.1%}, {n_correct}/{n_samples})"
        )

    return output
