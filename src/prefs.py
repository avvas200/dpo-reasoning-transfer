"""Preference-set utilities for the dpo-reasoning-transfer project.

This module builds the Condition B ("self-generated reasoning") preference
dataset described in the RESEARCH_PLAN. It also handles formatting pairs for
TRL's `DPOTrainer` so Notebook 04 has a single source of truth.

Design notes
------------
* We sample multiple responses per problem with `num_return_sequences` so
  the GPU sees a real batch rather than K sequential calls.
* Pair construction is deliberately simple: uniformly random `chosen` from
  the correct chains, uniformly random `rejected` from the incorrect chains.
  We do NOT bias toward "shortest correct / longest rejected", because that
  would introduce an anti-length signal that confounds the reasoning-signal
  hypothesis. The resulting length distributions are logged so any residual
  length effect can be discussed as a limitation.
* An answer is "correct" iff its extracted final number matches the GSM8K
  ground truth under `src.eval.answers_match`. This means a chain can be
  labeled correct while containing broken reasoning that happens to land on
  the right number ("spurious correctness"). This is an accepted limitation
  of pairwise preference from a final-answer verifier alone; more careful
  filtering (process rewards, verifier chains) is future work.
* All sampling is source-only from GSM8K **train**. The test split (which
  Notebook 01 evaluates on) is never touched here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm

from src.eval import GSM8K_PROMPT, answers_match, extract_answer


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_multiple_responses(
    model,
    tokenizer,
    question: str,
    K: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 384,
    seed: Optional[int] = None,
) -> list[str]:
    """Sample K stochastic chain-of-thought responses for one question.

    Uses `num_return_sequences=K` so all K chains are decoded in one GPU
    call. On a T4 with a 3B model and a ~100-token GSM8K question, this
    is meaningfully faster than K separate calls at the cost of ~K× the
    peak KV-cache memory (still well within 16 GB).

    If `seed` is given, `torch.manual_seed` is set before generation so the
    K responses are reproducible for that (question, seed) pair.
    """
    if seed is not None:
        torch.manual_seed(seed)

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
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=K,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        for seq in output
    ]


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_preference_pair(
    question: str,
    gt_answer: str,
    responses: list[str],
    rng: Optional[random.Random] = None,
) -> Optional[dict]:
    """Build a `(chosen, rejected)` pair from K sampled responses.

    A pair is producible iff at least one response is correct AND at least
    one is incorrect (final-answer match against `gt_answer`). Returns
    `None` otherwise — the caller decides whether to skip the problem or
    resample.

    The returned dict is the "raw" audit record: it includes the K
    predicted answers, which sample indices were chosen, and per-response
    lengths. It is NOT yet in DPOTrainer format — call
    `format_pair_for_dpo` for that.
    """
    if rng is None:
        rng = random.Random()

    correct: list[tuple[int, str]] = []
    incorrect: list[tuple[int, str]] = []
    all_answers: list[Optional[str]] = []
    for i, resp in enumerate(responses):
        pred = extract_answer(resp)
        all_answers.append(pred)
        if answers_match(pred, gt_answer):
            correct.append((i, resp))
        else:
            incorrect.append((i, resp))

    if not correct or not incorrect:
        return None

    chosen_idx, chosen_resp = rng.choice(correct)
    rejected_idx, rejected_resp = rng.choice(incorrect)

    return {
        "question": question,
        "gt_answer": gt_answer,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
        "chosen_answer": extract_answer(chosen_resp),
        "rejected_answer": extract_answer(rejected_resp),
        "chosen_length_chars": len(chosen_resp),
        "rejected_length_chars": len(rejected_resp),
        "chosen_sample_idx": chosen_idx,
        "rejected_sample_idx": rejected_idx,
        "n_correct_samples": len(correct),
        "n_incorrect_samples": len(incorrect),
        "all_sample_answers": all_answers,
    }


# ---------------------------------------------------------------------------
# DPO formatting (consumed by Notebook 04)
# ---------------------------------------------------------------------------

def format_pair_for_dpo(pair: dict, tokenizer) -> dict:
    """Convert one raw Condition B (self-generated reasoning) pair into
    TRL DPOTrainer's expected format.

    DPOTrainer expects three string fields:
        prompt   — everything up to and including the assistant turn opener
        chosen   — the preferred assistant response (no template wrapping)
        rejected — the dispreferred assistant response
    """
    prompt_messages = [
        {"role": "user", "content": GSM8K_PROMPT.format(question=pair["question"])}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": formatted_prompt,
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
    }


def format_ultrafeedback_example(example: dict, tokenizer) -> dict:
    """Convert one HuggingFaceH4/ultrafeedback_binarized row into DPOTrainer format.

    UltraFeedback stores `chosen`/`rejected` as lists of chat messages; we
    extract the last assistant turn's content and apply the tokenizer's chat
    template to the user prompt. This is the Condition A (general-purpose
    preferences) formatter, used by Notebook 03.
    """
    prompt = example["prompt"]
    chosen_response = example["chosen"][-1]["content"]
    rejected_response = example["rejected"][-1]["content"]

    prompt_messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": formatted_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_preference_jsonl(path: str) -> list[dict]:
    """Load a JSONL preference file into a list of dicts."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def load_completed_indices(path: str) -> set[int]:
    """Read a JSONL preference file and return the set of `problem_index`
    values already present. Used for resuming a partial sampling run."""
    done: set[int] = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "problem_index" in rec:
                done.add(rec["problem_index"])
    return done


def load_processed_indices(path: str) -> set[int]:
    """A separate log of problems we've *attempted*, whether or not they
    yielded a pair. Prevents re-sampling problems that produced no usable
    pair on a previous run."""
    done: set[int] = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(int(line))
                except ValueError:
                    continue
    return done


# ---------------------------------------------------------------------------
# Main sampling driver (used by Notebook 02)
# ---------------------------------------------------------------------------

def build_reasoning_preference_set(
    model,
    tokenizer,
    source_dataset,
    output_jsonl_path: str,
    processed_log_path: str,
    metadata_path: str,
    target_n_pairs: int = 500,
    max_source_problems: int = 700,
    K: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 384,
    seed: int = 42,
) -> dict:
    """Iterate over `source_dataset` (GSM8K train subset), sample K
    responses per problem, and append preference pairs to `output_jsonl_path`.

    Resumable: skips problems already listed in `processed_log_path` and
    stops early once `target_n_pairs` is reached.

    Returns a metadata dict that is also written to `metadata_path`.
    """
    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_indices(output_jsonl_path)
    processed = load_processed_indices(processed_log_path)
    n_pairs_start = len(completed)
    print(
        f"[prefs] Resuming with {n_pairs_start} pairs already saved "
        f"({len(processed)} problems previously attempted)."
    )

    # Opportunistic Unsloth fast-inference
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    rng = random.Random(seed)
    n_pairs = n_pairs_start
    n_attempted_this_run = 0

    pref_fh = open(output_jsonl_path, "a")
    proc_fh = open(processed_log_path, "a")
    try:
        pbar = tqdm(
            range(min(max_source_problems, len(source_dataset))),
            initial=len(processed),
            total=min(max_source_problems, len(source_dataset)),
            desc="Sampling",
        )
        for i in pbar:
            if i in processed:
                continue
            if n_pairs >= target_n_pairs:
                print(f"[prefs] Reached target of {target_n_pairs} pairs at "
                      f"source problem #{i}. Stopping.")
                break

            example = source_dataset[i]
            question = example["question"]
            gt_answer = extract_answer(example["answer"])

            responses = sample_multiple_responses(
                model, tokenizer, question,
                K=K, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed + i,   # per-problem seed for reproducibility
            )
            pair = build_preference_pair(question, gt_answer, responses, rng=rng)

            # Log attempt (whether or not it yielded a pair)
            proc_fh.write(f"{i}\n")
            proc_fh.flush()
            processed.add(i)
            n_attempted_this_run += 1

            if pair is not None:
                pair["problem_index"] = i
                pref_fh.write(json.dumps(pair) + "\n")
                pref_fh.flush()
                n_pairs += 1

            yield_pct = 100 * n_pairs / max(1, len(processed))
            pbar.set_postfix(
                pairs=n_pairs,
                yield_pct=f"{yield_pct:.1f}%",
            )
    finally:
        pref_fh.close()
        proc_fh.close()

    metadata = {
        "sampling_config": {
            "K": K,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        "source": {
            "dataset": "openai/gsm8k",
            "split": "train",
            "max_source_problems": max_source_problems,
        },
        "yield": {
            "n_processed": len(processed),
            "n_pairs": n_pairs,
            "yield_rate": n_pairs / len(processed) if processed else 0.0,
            "n_attempted_this_run": n_attempted_this_run,
        },
        "output": {
            "pairs_jsonl": str(output_jsonl_path),
            "processed_log": str(processed_log_path),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[prefs] Wrote metadata to {metadata_path}")

    return metadata
