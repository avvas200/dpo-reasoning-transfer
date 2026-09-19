# Research Plan
## When does preference optimization transfer to reasoning?
### A preference-signal ablation on Llama-3.2-3B-Instruct

---

## 1. Motivation and preliminary result

Direct Preference Optimization (DPO; Rafailov et al., 2023) has become the default post-training recipe for aligning language models to human preferences, replacing the more expensive PPO-based RLHF pipeline. It is empirically effective at improving conversational quality on chat benchmarks. Whether that benefit extends to *specialized capabilities* — mathematical reasoning, coding, factual recall — is less clear. Prior work has documented an "alignment tax" (Askell et al., 2021; Ouyang et al., 2022) in which alignment training on general instruction-following data degrades performance on capability benchmarks, but the interaction is not well characterized for DPO specifically at the small-model / small-data scale that is common in practice.

Our preliminary experiment applied DPO to `Llama-3.2-3B-Instruct` using 500 preference pairs sampled from the standard `HuggingFaceH4/ultrafeedback_binarized` dataset, then evaluated on the first 100 problems of the GSM8K test set. The result was 59.0% → 60.0% accuracy — a single problem flipped from incorrect to correct, no problems flipped in the other direction, and an exact McNemar test on the discordant pairs gives p = 1.0. Mean response length and formatting were essentially unchanged (640 → 639 characters). The DPO adapters, at this scale and with this preference signal, are a near no-op on this benchmark.

This is a *defensible* observation but not, on its own, a research contribution. Three things are missing:
1. **Statistical power.** With n = 100 and a single-proportion 95% CI of roughly ±9.6 percentage points, we cannot distinguish a 1 pp swing from noise.
2. **A hypothesis worth testing.** Reporting a null result on one preference set answers no interesting scientific question.
3. **A mechanism story.** Accuracy alone tells us nothing about *what changed* inside the model — an important question if the null accuracy result masks a real distributional shift.

This plan extends the preliminary experiment along all three axes.

---

## 2. Research question and hypotheses

**Central question.** *Under what conditions does preference optimization transfer to specialized mathematical reasoning?*

We treat "the alignment tax on reasoning" not as a fixed property of DPO, but as a function of the *preference signal* being optimized against. We ablate the signal while holding everything else constant.

**H1 (null-result replication at scale).** DPO on general-purpose preference data (UltraFeedback) does not significantly change GSM8K accuracy across seeds. *Predicted effect:* |Δ| < 2 pp, paired-bootstrap 95% CI includes zero, McNemar p > 0.05.

**H2 (main hypothesis).** DPO on a self-generated preference set — where `chosen` is a correct chain-of-thought sampled from the base model and `rejected` is an incorrect chain-of-thought on the same problem — produces a positive, statistically significant improvement on GSM8K. *Predicted effect:* +3 to +8 pp mean improvement across seeds, McNemar p < 0.05 on the pooled runs.

**H3 (mechanistic).** The two DPO variants exhibit distinct token-distribution shifts (measured by KL divergence from the base model on held-out problems), even in the cells where their accuracies converge or diverge. Specifically: the general-preference variant will show diffuse, low-magnitude shifts across most tokens; the reasoning-preference variant will show larger shifts concentrated at *decision points* (equation setup, intermediate arithmetic).

**Falsification.** If H2 fails (self-generated reasoning preferences also produce no significant improvement), the conclusion is that the reasoning alignment tax at this model scale is *robust* to preference-signal engineering — itself a publishable observation, and one that motivates the interpretation that reasoning requires signal beyond pairwise preferences (e.g., process rewards, verifier feedback).

---

## 3. Method

### 3.1 Base model and infrastructure

- **Model:** `unsloth/Llama-3.2-3B-Instruct` in 4-bit quantization (NF4).
- **Adapters:** LoRA with r = 16, α = 16, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Stack:** Unsloth + TRL's `DPOTrainer` on a single T4 (Colab free-tier).
- **Reference model:** The frozen 4-bit base (Unsloth's QLoRA-DPO shortcut — no second model loaded).

### 3.2 Preference data conditions

**Condition A — General-purpose preferences (control).**
`HuggingFaceH4/ultrafeedback_binarized`, `train_prefs` split, shuffled with fixed seed, first 500 pairs. This is the same dataset used by the original DPO paper and Zephyr (Tunstall et al., 2023).

**Condition B — Self-generated reasoning preferences (treatment).**
Constructed as follows:
1. Sample 4 chains-of-thought per problem from the base model on GSM8K *train* set (temperature = 0.8, top-p = 0.95). Cap at ~1,000 problems until we have 500 usable pairs.
2. For each problem, extract the final numeric answer from each chain.
3. Retain problems where *both* a correct and an incorrect chain were generated.
4. `chosen` = correct chain, `rejected` = incorrect chain. If multiple candidates per side exist, take the shortest correct and the longest incorrect (a mild filter for signal quality; the length bias is documented and reported).
5. Save the resulting dataset to `data/self_generated_prefs.jsonl` with metadata (source problem index, seed, chain lengths, extracted answers).

Both conditions have identical size (500 pairs), identical prompt-template formatting, and identical DPO hyperparameters.

### 3.3 DPO training recipe (identical for both conditions)

| Hyperparameter | Value | Source |
|---|---|---|
| β | 0.1 | DPO paper default |
| Learning rate | 5e-6 | Zephyr recipe |
| Batch size (effective) | 8 | 2 × 4 gradient accumulation |
| Epochs | 1 | DPO overfits fast |
| Warmup ratio | 0.1 | Zephyr recipe |
| Optimizer | AdamW-8bit | Unsloth default |
| Max sequence length | 1024 | Fits T4 memory |
| Seeds | {1, 2, 3} | Report mean ± sd |

### 3.4 Evaluation protocol

- **Benchmark:** GSM8K test set, first 500 problems (5× the preliminary experiment). We fix the subset across all runs so pairwise tests are valid.
- **Prompting:** Chat template, one-shot user turn: `"Solve this problem step by step. End your response with the final numeric answer."`
- **Decoding:** Greedy (do_sample = False), max_new_tokens = 512.
- **Answer extraction:** Ground truth is the number after `####`. Model answer is the last number in the generated text (standard "strict match" convention). Both stripped of commas.
- **Correctness:** Exact string match after stripping.
- **Cached artifacts:** Full per-problem generations (question, gt_answer, pred_answer, response, correct) saved to JSON for every run.

This is deliberately the standard community protocol — no bespoke tricks — so the results are directly comparable to the surrounding literature.

---

## 4. Statistical protocol

We commit to the following analyses *before* running the treatment condition:

1. **Primary test (per seed):** McNemar's exact test on the discordant pairs between Condition B and baseline. Alpha = 0.05.
2. **Aggregated test:** Cochran's Q across seeds, or pooled McNemar if seed-heterogeneity is low.
3. **Effect size:** Paired difference in accuracy, with a 95% percentile bootstrap CI (10,000 resamples of the 500 problems).
4. **Multiple comparisons:** Two families of tests (A vs. baseline; B vs. baseline). Bonferroni-adjusted α = 0.025 for the primary claims.
5. **Report:** mean ± sd across seeds for both conditions, alongside the individual per-seed p-values.

**What counts as "H2 supported":** mean Δ ≥ +3 pp across the 3 seeds *and* pooled McNemar p < 0.025 *and* baseline-vs-B bootstrap CI excludes zero. Anything weaker is reported honestly as "trend but underpowered."

---

## 5. Experimental matrix and compute budget

| Run | Condition | Seed | Training time | Eval time | Cache key |
|---|---|---|---|---|---|
| 0 | Baseline | — | 0 | ~1.5 h | `baseline_gsm8k_500.json` |
| 1 | A (UltraFeedback) | 1 | ~30 min | ~1.5 h | `dpo_A_seed1.json` |
| 2 | A | 2 | ~30 min | ~1.5 h | `dpo_A_seed2.json` |
| 3 | A | 3 | ~30 min | ~1.5 h | `dpo_A_seed3.json` |
| 4 | B (self-gen prefs) | 1 | ~30 min | ~1.5 h | `dpo_B_seed1.json` |
| 5 | B | 2 | ~30 min | ~1.5 h | `dpo_B_seed2.json` |
| 6 | B | 3 | ~30 min | ~1.5 h | `dpo_B_seed3.json` |
| — | Build B dataset | — | ~1 h | — | `data/self_generated_prefs.jsonl` |

**Total:** ~14 hours of T4 wall-clock. Split across ~4 Colab sessions of ≤ 3.5 hours each (well below the ~4 h idle-disconnect threshold that has been common on the free tier). Every artifact is written to Google Drive after each session so no work is lost.

**Colab survival strategy.**
- Mount Drive at the top of every notebook; persist model outputs, checkpoints, and JSON there.
- Every notebook must be resumable: if it crashes at problem 231/500, the next run picks up at 231.
- Keep browser tab active during long evals; use a small JS keepalive if needed (documented in the notebook).
- Baseline eval runs *once* and is cached forever.

---

## 6. Notebooks

Each notebook is self-contained, parameterized by seed where applicable, and reads/writes to a shared `results/` directory on Drive.

1. **`00_setup_and_smoke_test.ipynb`** (~10 min)
   Install pinned deps, load the base model, run 5-problem sanity check, verify Drive mount.

2. **`01_baseline_eval.ipynb`** (~1.5 h, run once)
   Evaluate the untouched instruct model on GSM8K[:500]. Save `results/baseline_gsm8k_500.json`.

3. **`02_build_reasoning_prefs.ipynb`** (~1 h, run once)
   Sample 4 CoTs per problem from base model on GSM8K train, filter to (correct, incorrect) pairs, save 500 pairs to `data/self_generated_prefs.jsonl`. Also save yield statistics (how many problems produced usable pairs, distribution of chain lengths).

4. **`03_dpo_A_ultrafeedback.ipynb`** (~2 h per seed, 3 seeds)
   DPO on Condition A, evaluate immediately, save `results/dpo_A_seed{k}.json`. Parameterized by `SEED` cell at the top.

5. **`04_dpo_B_reasoning.ipynb`** (~2 h per seed, 3 seeds)
   Same as (4) but Condition B. Saves `results/dpo_B_seed{k}.json`.

6. **`05_statistics_and_figures.ipynb`** (~15 min, CPU-only)
   Reads all JSON results. Runs McNemar, bootstrap CIs, produces the main results figure, the per-condition accuracy distribution, and a discordant-pairs breakdown table.

7. **`06_mechanistic_analysis.ipynb`** (~45 min)
   For a held-out subset of 50 GSM8K problems: compute per-token KL divergence between base and each DPO model. Aggregate by problem region (question echo vs. reasoning vs. final answer). Plot LoRA weight-norm changes per module. Sample 3 discordant problems for qualitative narrative in the report.

8. **`07_mllm_extension_STRETCH.ipynb`** (~4 h, optional)
   Only if 01–06 are clean and time permits. Replicate the A-vs-B protocol on `Qwen/Qwen2-VL-2B-Instruct` evaluated on MathVista-mini (1000 items or a subsample). This is the direct hook to Prof. Huang's MLLM interest but carries significant Colab risk.

---

## 7. Mechanistic analysis (Notebook 6, expanded)

Three cheap-but-informative probes:

**(a) Token-level KL divergence.** For each held-out problem, decode with the base model, then compute `KL(p_DPO || p_base)` at each generated token position using the base model's tokens as the reference sequence (i.e., teacher-forced). Average over problems, plot as a function of relative position within the response.

**(b) LoRA weight-norm audit.** For each LoRA-targeted module, report the Frobenius norm of `B·A` — the effective weight delta the adapter applies. Compare Condition A vs. Condition B module-by-module. A striking finding would be that B concentrates more mass in later MLP layers (associated with computation) while A spreads it thin.

**(c) Discordant-problem coding.** For every problem where any DPO run's answer differs from baseline's answer, manually code the error type: {arithmetic-slip, unit-error, misread-problem, correct-approach-wrong-answer, correct}. Report as a stacked bar. This is qualitative but grounding: it tells a reader *what actually changed*.

---

## 8. Stretch: MLLM extension (Notebook 7)

**Motivation for Prof. Huang's group.** His stated interests include Multimodal LLMs and Vision-Language Models. Showing that the *same experimental protocol* generalizes (or fails to generalize) from text-only reasoning to visual reasoning is a direct proof-of-competence in the MLLM setting and a natural bridge into his group's problems.

**Design.** `Qwen/Qwen2-VL-2B-Instruct` (fits in 4-bit on T4). Evaluate on MathVista `testmini` (1000 items). Condition A = general MLLM preference data (e.g., subset of `openbmb/RLAIF-V-Dataset` or similar). Condition B = self-generated MathVista pairs. 1 seed only under compute constraints; explicitly note this as a preliminary probe, not a full replication.

**Risk.** Real chance the VLM evaluation exceeds Colab session limits. If it does, we report partial results honestly and treat it as future work.

---

## 9. Fit to the target research group

Prof. Huang's listed interests (LLM/MLLM/VLM; SFT/RL; data synthesis; AI agents; embodied AI; AI + X) intersect this work at three points:

- **Post-training methods (SFT/RL).** DPO sits squarely in the RL-free preference optimization family. The ablation here is a direct study of *what makes preference signals effective*, which is a live question for anyone doing post-training research.
- **Data synthesis.** Condition B *is* a data synthesis pipeline (self-generation of contrastive reasoning chains, filtered by verifier). The report should frame this explicitly.
- **MLLM (via Notebook 7 stretch).** The extension pattern demonstrates comfort with the modality Prof. Huang works in most.

The report and README will position the work honestly: a small, single-model, T4-scale empirical study of one hypothesis, contributed as a demonstration of research methodology (question framing, pre-registered protocol, honest statistics, mechanistic follow-up) rather than as a state-of-the-art claim.

---

## 10. Repository structure

```
dpo-reasoning-transfer/
├── README.md                 # 2-page project overview, TL;DR, results, links
├── REPORT.md                 # 8-10 page write-up (the actual deliverable to cite in SOP)
├── RESEARCH_PLAN.md          # This file — commits us to the protocol before results
├── LICENSE
├── requirements.txt          # Pinned versions
├── .gitignore
├── notebooks/
│   ├── 00_setup_and_smoke_test.ipynb
│   ├── 01_baseline_eval.ipynb
│   ├── 02_build_reasoning_prefs.ipynb
│   ├── 03_dpo_A_ultrafeedback.ipynb
│   ├── 04_dpo_B_reasoning.ipynb
│   ├── 05_statistics_and_figures.ipynb
│   ├── 06_mechanistic_analysis.ipynb
│   └── 07_mllm_extension_STRETCH.ipynb
├── src/
│   ├── eval.py               # GSM8K eval loop, shared by all notebooks
│   ├── prefs.py              # Preference dataset construction and formatting
│   ├── stats.py              # McNemar, bootstrap CI, aggregation
│   └── plots.py              # Consistent matplotlib style
├── data/
│   ├── self_generated_prefs.jsonl
│   └── README.md             # Data card: how it was made, size, license
├── results/
│   ├── baseline_gsm8k_500.json
│   ├── dpo_A_seed{1,2,3}.json
│   ├── dpo_B_seed{1,2,3}.json
│   ├── mechanistic/
│   └── figures/
│       ├── main_results.png
│       ├── kl_by_position.png
│       ├── weight_norm_deltas.png
│       └── error_type_shift.png
└── docs/
    └── colab_survival_guide.md   # For anyone reproducing on free-tier
```

---

## 11. Report structure

The eventual `REPORT.md` will follow a standard short-paper skeleton:

1. **Abstract** (~150 words)
2. **Introduction** — the alignment-tax question, our contribution in one sentence
3. **Related work** — DPO, alignment tax, self-play / self-generated preference data (STaR, ReST, RLAIF-V), verifier-based approaches
4. **Method** — model, two preference conditions, DPO recipe, evaluation
5. **Experiments** — statistical protocol, baseline and both conditions across seeds
6. **Mechanistic analysis** — KL divergence, weight norms, error typology
7. **Discussion** — what supports/refutes each hypothesis, implications for post-training design
8. **Limitations** — single model family, single benchmark family, T4-scale training data, no comparison to RLHF or other objectives
9. **Reproducibility statement** — pinned deps, seeds, cached artifacts, one-click Colab links
10. **References**

---

## 12. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Colab session drops mid-run | High | Resumable eval loop; per-problem checkpointing to Drive. |
| GPU quota exhausted for the day | Medium | Space runs across days; batch-plan the sessions. |
| Condition B yields fewer than 500 usable pairs | Medium | Expand source pool from GSM8K train (7473 problems available); increase samples-per-problem to 8. |
| DPO training diverges | Low | β = 0.1 and lr = 5e-6 are known-safe. Monitor `rewards/margins`. |
| VLM extension infeasible on T4 | High | Explicitly labeled stretch; drop cleanly if it doesn't fit. |
| H2 not supported (no significant improvement) | Medium | Plan is honest either way; H1 + null H2 + mechanistic story is still a publishable narrative. |
| Random-baseline drift on GSM8K subset | Low | Same fixed subset used for every eval; no re-sampling. |

---

## 13. Timeline (realistic part-time)

- **Week 1:** Notebooks 00, 01, 02. Cache baseline. Build Condition B dataset.
- **Week 2:** Notebooks 03, 04. Run all 6 DPO experiments (3 seeds × 2 conditions).
- **Week 3:** Notebooks 05, 06. Statistics, figures, mechanistic analysis. Draft README + REPORT.
- **Week 4:** Polish. Attempt Notebook 07 (stretch). Push clean repo to GitHub with README, tagged release.

Total: 4 weeks part-time, or ~2 weeks if daily focused work.

---

## 14. References to include (verify each before final commit)

The following references belong in the report. I've listed them from memory; the student should verify title, author list, venue, and year for each before citing.

- Rafailov, R. et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
- Ouyang, L. et al. *Training language models to follow instructions with human feedback.* NeurIPS 2022. (InstructGPT — original alignment-tax observation.)
- Askell, A. et al. *A general language assistant as a laboratory for alignment.* arXiv 2021. (Alignment-tax terminology.)
- Cobbe, K. et al. *Training verifiers to solve math word problems.* arXiv 2021. (GSM8K.)
- Cui, G. et al. *UltraFeedback: Boosting Language Models with High-quality Feedback.* 2023. (UltraFeedback dataset.)
- Tunstall, L. et al. *Zephyr: Direct Distillation of LM Alignment.* 2023. (DPO recipe reference.)
- Zelikman, E. et al. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS 2022. (Self-generated reasoning traces.)
- Lightman, H. et al. *Let's Verify Step by Step.* ICLR 2024. (Process supervision — contextualizes why pairwise preference may be too coarse.)
- Yu, T. et al. *RLAIF-V: Aligning MLLMs through Open-Source AI Feedback.* 2024. (Relevant if Notebook 7 runs.)
- Lu, P. et al. *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts.* ICLR 2024. (Relevant if Notebook 7 runs.)

I do not have web access to verify these here; every citation must be double-checked against the actual paper before the report is submitted.

---

## Summary

This plan converts a null-result tutorial into a small but genuine empirical study by (a) restoring statistical power through larger eval sets and multiple seeds, (b) posing a specific, falsifiable hypothesis about *which preference signals* transfer to reasoning, and (c) adding mechanistic follow-up to characterize what changes inside the model regardless of accuracy outcomes. It fits within Colab free-tier compute if executed carefully. The narrative works whether H2 is supported or refuted. It positions the student honestly for HKPFS review by demonstrating research methodology aligned to Prof. Huang's stated interests in post-training, data synthesis, and MLLM.
