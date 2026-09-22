# Titans: A Small-Scale Reproduction Study

Reproducing key results from **"Titans: Learning to Memorize at Test Time"**
(Behrouz et al., Google Research, 2025) under free-tier compute (a single
Google Colab T4 GPU).

Course project for **Speech and Natural Language Processing**, School of ECE,
National Technical University of Athens (NTUA).

---

## TL;DR

We reproduce the Titans architecture at ~1000× smaller scale than the original
paper and map out **which of its claims survive under a tiny compute budget**.

| Experiment | Task | Result |
|---|---|---|
| **I — Retrieval** | Needle-in-a-Haystack | ✅ Positive — MAC holds the needle up to 4× its training context |
| **II — Reasoning** | Multi-hop QA (BABILong-style) | ❌ Negative — MAC stays at chance; needs billion-token pretraining |
| **III — Forecasting** | ETTh1 long-term forecasting | ✅ Positive — memory-only LMM overtakes the Transformer at long horizons |

The overarching finding: the neural memory helps exactly when information must
survive **beyond the attention window** — but *how* that ability is trained
matters. Tasks that only need to **compress continuous structure** (forecasting)
work at small scale; tasks that need **discrete cross-segment retrieval learned
from scratch** (reasoning) do not.

---

## Repository structure

- `paper/` — the write-up (PDF)
- `notebooks/` — runnable Colab notebooks, one per experiment
- `src/` — the same code as standalone Python scripts
- `figures/` — generated plots used in the paper

---

## Experiments

### Experiment I — Needle-in-a-Haystack (retrieval)
A Baseline Transformer and a parameter-matched MAC Titan (~12.6M params) are
trained on WikiText with a 1K context and evaluated on retrieval across context
lengths (1K–16K) and needle depths. The MAC Titan retrieves perfectly up to 4×
its training context, where the plain Transformer already fails.

### Experiment II — Multi-hop reasoning (negative result)
BABILong-style QA: one or two supporting facts are hidden in noise and the model
must combine them to answer. The MAC Titan never rises above chance
(loss plateaus at exactly `ln(6) ≈ 1.79`), while the Transformer solves the task.
We rule out trivial bugs (data leakage, poisoned init, LR-scheduler miscount,
over-hard objective) and attribute the failure to an **architectural reality**:
MAC must learn cross-segment retrieval through its gradient inner loop, which
requires pretraining on billions of tokens (the paper uses 15–30B tokens and
170–760M-param models; our budget is ~2M tokens). Consistent with the
independent *Titans Revisited* (Di Nepi et al., 2025).

A synthetic S-NIAH probe (`experiment1_niah_synthetic.ipynb`) reproduces the
same collapse in a clean, controlled setting, confirming the cause is scale,
not implementation.

### Experiment III — Long-term time-series forecasting (positive result)
The standalone Long-term Memory Module (LMM) replaces Mamba in a Simba-style
pipeline on **ETTh1** (channel-independent patching, input length 512, horizons
96/192/336/720h). The LMM's advantage grows monotonically with the horizon and
**overtakes the Transformer at 720h** (MSE 0.635 vs 0.666, +4.7%),
qualitatively reproducing the paper's long-term-memory claim.

---

## Setup

```bash
pip install -r requirements.txt
```

Main dependencies: `titans-pytorch`, `torch`, `transformers`, `numpy`,
`matplotlib`. Datasets: WikiText-2 (reasoning), a synthetic generator (NIAH),
and [ETTh1](https://github.com/zhouhaoyi/ETDataset) (forecasting).

Each notebook is self-contained and runs top-to-bottom on a free Colab T4 GPU.

---

## Notes

This is a **teaching reproduction**, not an official implementation. Absolute
numbers differ from the paper (smaller models, fewer epochs); the goal is to
reproduce the *qualitative* behaviour and to honestly document where small-scale
reproduction breaks down.

## References

- Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* 2025.
- Di Nepi et al. *Titans Revisited.* 2025.
- [`titans-pytorch`](https://github.com/lucidrains/titans-pytorch) (lucidrains)
