"""DPO training driver for the dpo-reasoning-transfer project.

Both DPO notebooks (03_dpo_A_ultrafeedback and 04_dpo_B_reasoning) call
`train_dpo_and_evaluate` from here. Keeping the training loop in one place
means the two conditions cannot silently diverge on hyperparameters — the
only difference between them is the preference dataset.

Import ordering note
--------------------
Unsloth's `PatchDPOTrainer()` must be called BEFORE `trl.DPOTrainer` is
imported anywhere in the process. We therefore defer all TRL imports into
the functions below, and invoke `PatchDPOTrainer()` first in every call
site. This lets `import src.train` at the top of a notebook be safe
regardless of when the notebook eventually decides to train.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------

# Fixed target modules — the standard "attention + MLP" set for Llama-style
# models, matching the recipe used in the preliminary run and in Zephyr.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def attach_lora(
    model,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    seed: int = 3407,
):
    """Wrap the base model with LoRA adapters for DPO training.

    Kept identical across the two DPO conditions so the only experimental
    variable is the preference dataset.
    """
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=LORA_TARGETS,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # required to fit on T4
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )


# ---------------------------------------------------------------------------
# DPO training
# ---------------------------------------------------------------------------

# Frozen hyperparameters, matching the preliminary run and the RESEARCH_PLAN.
# Exposed as kwargs but not typically changed between runs.
DEFAULT_DPO_HPARAMS = dict(
    beta=0.1,
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch size 8
    warmup_ratio=0.1,
    max_length=1024,
    max_prompt_length=512,
    weight_decay=0.0,
    lr_scheduler_type="linear",
    logging_steps=10,
)


def train_dpo(
    model,
    tokenizer,
    dpo_dataset,
    seed: int = 42,
    output_dir: str = "/tmp/dpo_out",
    hparams: Optional[dict] = None,
):
    """Train DPO for one epoch on `dpo_dataset`.

    Returns `(trainer_stats, log_history)` where `log_history` is a list of
    the per-step scalars TRL logs (loss, rewards/margins, etc.), suitable
    for JSON-serialising into a run's metadata block.
    """
    # PatchDPOTrainer must precede DPOTrainer import.
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()

    import torch
    from trl import DPOConfig, DPOTrainer

    hp = {**DEFAULT_DPO_HPARAMS, **(hparams or {})}

    dpo_config = DPOConfig(
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        warmup_ratio=hp["warmup_ratio"],
        num_train_epochs=hp["num_train_epochs"],
        learning_rate=hp["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=hp["logging_steps"],
        optim="adamw_8bit",
        weight_decay=hp["weight_decay"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        seed=seed,
        output_dir=output_dir,
        max_length=hp["max_length"],
        max_prompt_length=hp["max_prompt_length"],
        beta=hp["beta"],
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,        # Unsloth QLoRA shortcut: ref = frozen base adapters
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer_stats = trainer.train()
    training_seconds = time.time() - t0

    log_history = list(trainer.state.log_history) if hasattr(trainer, "state") else []

    return trainer_stats, log_history, training_seconds, hp


# ---------------------------------------------------------------------------
# End-to-end: train, evaluate, save
# ---------------------------------------------------------------------------

def train_dpo_and_evaluate(
    model,
    tokenizer,
    dpo_dataset,
    results_output_path: str,
    condition_name: str,
    preference_dataset_name: str,
    seed: int = 42,
    n_eval_samples: int = 500,
    output_dir: str = "/tmp/dpo_out",
    hparams: Optional[dict] = None,
    force_rerun: bool = False,
    adapter_save_path: Optional[str] = None,
) -> dict:
    """The end-to-end call both DPO notebooks make.

    Skips work if the results file already exists (unless `force_rerun`),
    so re-running a completed seed is instant. Otherwise: attach nothing
    (the caller has already attached LoRA), train DPO, then evaluate on
    GSM8K[:n_eval_samples] and save to `results_output_path`.

    Args:
        model, tokenizer: HF-compatible model and tokenizer, already wrapped
            with LoRA adapters via `attach_lora`.
        dpo_dataset: A `datasets.Dataset` with columns `prompt`, `chosen`,
            `rejected` (strings).
        results_output_path: Where to write the eval JSON. Convention:
            `.../results/dpo_{A,B}_seed{seed}.json`.
        condition_name: "dpo_A_ultrafeedback" or "dpo_B_reasoning".
        preference_dataset_name: Free-form identifier for provenance
            (dataset URL, local jsonl path, etc.).
        seed: Training seed. Also embedded into the results JSON.
        n_eval_samples: How many GSM8K test problems to evaluate on.
        output_dir: Scratch dir for the trainer.
        hparams: Optional overrides to `DEFAULT_DPO_HPARAMS`.
        force_rerun: If True, ignore any existing results and retrain + re-eval.
        adapter_save_path: If given, save the trained LoRA adapter here after
            training (before eval). Notebook 06 loads these to compute KL
            divergence and weight-norm statistics. Backwards-compatible —
            skipped if None.

    Returns:
        The results dict (as written to `results_output_path`).
    """
    from src.eval import evaluate_gsm8k

    # --- Cache short-circuit ---------------------------------------------
    if not force_rerun and Path(results_output_path).exists():
        try:
            with open(results_output_path) as f:
                cached = json.load(f)
            if (
                cached.get("n_samples") == n_eval_samples
                and cached.get("run_metadata", {}).get("seed") == seed
                and cached.get("run_metadata", {}).get("condition") == condition_name
            ):
                print(
                    f"[train] Cached results for {condition_name} seed={seed}: "
                    f"{results_output_path}"
                )
                print(f"[train] Accuracy: {cached['accuracy']:.1%}")
                print("[train] Skipping training + eval. "
                      "Pass force_rerun=True to override.")
                return cached
            else:
                print(f"[train] Cached file exists but does not match "
                      f"(seed/condition/n_samples). Retraining.")
        except (json.JSONDecodeError, KeyError):
            print(f"[train] Cached file unreadable; retraining.")

    # --- Train -----------------------------------------------------------
    print(f"[train] Training DPO: condition={condition_name}, seed={seed}, "
          f"n_pairs={len(dpo_dataset)}")
    trainer_stats, log_history, training_seconds, hp = train_dpo(
        model, tokenizer, dpo_dataset,
        seed=seed, output_dir=output_dir, hparams=hparams,
    )
    print(f"[train] DPO finished in {training_seconds:.0f}s.")

    # Save the LoRA adapter for downstream mechanistic analysis (Notebook 06).
    if adapter_save_path is not None:
        import os
        os.makedirs(adapter_save_path, exist_ok=True)
        model.save_pretrained(adapter_save_path)
        # Also drop a small provenance file so Notebook 06 can verify
        # what it's loading.
        with open(os.path.join(adapter_save_path, "provenance.json"), "w") as f:
            json.dump({
                "condition": condition_name,
                "seed": seed,
                "preference_dataset": preference_dataset_name,
                "training_seconds": training_seconds,
                "n_pairs": len(dpo_dataset),
                "hparams": hp,
            }, f, indent=2)
        print(f"[train] Adapter saved to {adapter_save_path}")

    # Extract a compact training summary for the JSON
    final_loss = None
    final_margin = None
    for entry in reversed(log_history):
        if final_loss is None and "loss" in entry:
            final_loss = entry["loss"]
        if final_margin is None and "rewards/margins" in entry:
            final_margin = entry["rewards/margins"]
        if final_loss is not None and final_margin is not None:
            break

    training_summary = {
        "training_seconds": training_seconds,
        "n_pairs": len(dpo_dataset),
        "final_loss": final_loss,
        "final_rewards_margin": final_margin,
        "hparams": hp,
        # Full log history is small (~50 entries × ~6 scalars); include it.
        "log_history": log_history,
    }

    # --- Evaluate --------------------------------------------------------
    print(f"[train] Evaluating on GSM8K[:{n_eval_samples}]...")
    results = evaluate_gsm8k(
        model, tokenizer,
        n_samples=n_eval_samples,
        output_path=results_output_path,
        run_metadata={
            "condition": condition_name,
            "seed": seed,
            "base_model": "unsloth/Llama-3.2-3B-Instruct",
            "quantization": "4-bit",
            "preference_dataset": preference_dataset_name,
            "training": training_summary,
        },
        force_rerun=force_rerun,
    )

    print(f"[train] Done. Accuracy: {results['accuracy']:.1%} "
          f"({results['n_correct']}/{results['n_samples']})")
    return results


# ---------------------------------------------------------------------------
# Quick McNemar helper (used by Notebooks 03/04 for a fast preview
# against baseline). The full statistics story lives in Notebook 05.
# ---------------------------------------------------------------------------

def mcnemar_pvalue(b: int, c: int) -> float:
    """Two-tailed exact McNemar p-value on discordant pairs.

    b: # problems where baseline was correct and treatment was wrong.
    c: # problems where baseline was wrong and treatment was correct.
    Uses the exact binomial null (each discordant pair equally likely to
    go either way).
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    tail = sum(comb(n, k) for k in range(x + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare_to_baseline(baseline_results: dict, dpo_results: dict) -> dict:
    """Quick paired diagnostic between two eval JSONs on the same problem set.

    Returns delta accuracy, discordant-pair counts, and a McNemar p-value.
    Assumes the two runs used the same `n_samples` and problem indices.
    """
    b_results = {r["index"]: r for r in baseline_results["results"]}
    d_results = {r["index"]: r for r in dpo_results["results"]}
    common_ix = sorted(set(b_results) & set(d_results))

    b_right_d_wrong = 0
    b_wrong_d_right = 0
    both_right = 0
    both_wrong = 0
    for i in common_ix:
        b_ok = b_results[i]["correct"]
        d_ok = d_results[i]["correct"]
        if b_ok and not d_ok:
            b_right_d_wrong += 1
        elif not b_ok and d_ok:
            b_wrong_d_right += 1
        elif b_ok and d_ok:
            both_right += 1
        else:
            both_wrong += 1

    return {
        "n_compared": len(common_ix),
        "baseline_accuracy": baseline_results["accuracy"],
        "dpo_accuracy": dpo_results["accuracy"],
        "delta_pp": (dpo_results["accuracy"] - baseline_results["accuracy"]) * 100,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "baseline_right_dpo_wrong": b_right_d_wrong,
        "baseline_wrong_dpo_right": b_wrong_d_right,
        "mcnemar_p": mcnemar_pvalue(b_right_d_wrong, b_wrong_d_right),
    }
