# grokzip

**Not All Compression Is Equal: Rank Separates Memorisation from Generalisation in Grokking**

Project for _Deep Learning and Applied AI_, a.y. 2025/2026, Sapienza University of Rome.
Giovanni Ciarravano — `ciarravano.1989209@studenti.uniroma1.it`

---

## The question

During grokking a network reaches perfect training accuracy thousands of steps before it
generalises. In that interval the training signal is saturated and says nothing about
whether the network found an algorithm or a lookup table. Telling the two apart normally
requires a held-out set.

This project asks whether **post-training compression** can do it instead. The intuition:
a lookup table needs every entry stored precisely, whereas a rule tolerates perturbation.
If so, compressing the weights and re-running the _training_ data is enough — no held-out
labels, no prior knowledge of the circuit the network is supposed to have found.

Three compression operators are applied post hoc, at matched points on the trajectory,
with no retraining:

- **SVD rank truncation** to rank `r` (optimal rank-`r` approximation, Eckart–Young)
- **magnitude pruning** at sparsity `s` (one-shot, global)
- **uniform quantisation** at `b` bits (symmetric, per-tensor, absmax scale)

---

## Findings

### 1. Only rank separates the two phases

Retained training accuracy at each operator's reported operating point, over 12
generalising and 16 memorising anchors (mean ± s.d. across runs):

| operator      | generalising     | memorising   | gap       |
| ------------- | ---------------- | ------------ | --------- |
| **rank r=16** | **1.000 ± .000** | 0.137 ± .115 | **+0.86** |
| prune s=0.40  | 0.783 ± .185     | 0.402 ± .260 | +0.38     |
| quant. 3 bit  | 0.450 ± .186     | 0.229 ± .130 | +0.22     |

At rank 16 _every_ generalising anchor retains its full training accuracy, with zero
variance across runs. Pruning and quantisation separate two to four times less, with
overlapping error bands.

### 2. Three disjoint populations, separated without a test set

The **minimal viable rank** is defined on training data only:

```
r*(θ) = min { r : acc_train(trunc_r(θ)) ≥ 0.9 · acc_train(θ) }
```

Tracked across 2,564 checkpoints from 16 runs, it shows **two distinct descents**. The
first, from ≈70 to ≈35, happens even without generalisation (`nc_mem` makes it) but not
without weight decay (`nc_wd0` does not). The second, below ≈32, happens only under
generalisation.

| condition | data | weight decay | runs | final r\* | trajectory min r\* | detected |
| --------- | ---- | ------------ | ---- | --------- | ------------------ | -------- |
| `main`    | 30%  | 1.0          | 8    | 10–15     | 9–11               | 8/8      |
| `long`    | 25%  | 1.0          | 4    | 10–14     | 9–10               | 4/4      |
| `nc_mem`  | 20%  | 1.0          | 2    | 33–34     | 32                 | 0/2      |
| `nc_wd0`  | 30%  | 0.0          | 2    | 77–83     | 61–62              | 0/2      |

The controls are complementary. `nc_wd0` rules out long training as the cause;
`nc_mem` carries the _same_ weight decay as `main`, so any signal that is a by-product
of weight shrinkage would have to appear there too. Neither ever enters the generalising
band, at any point in training.

**Detector.** `t_detect = min { t : r*(t) ≤ R_crit }`, with `R_crit` fixed in advance as
the midpoint between the largest final r* (15) and the smallest plateau r* (35) over
`main` alone — giving **R_crit = 25**, with a margin of 10 to the worst positive and 7 to
the control minimum. Applied unchanged to `long` and the controls.

Calibration uses the labels of `main`; the detector itself does not. `long` is a
different experimental condition (25% of data, grokking at 13–26k steps instead of
4–9k), so transferring the threshold there unchanged is out-of-condition validation.

### 3. It does not fire early — a negative result

With `t_grok` the first step at test accuracy ≥ 0.5:

- median lead time **−130 steps**, mean −436
- **9 of 12** runs negative
- exact two-sided sign test on the paired design: **p = 0.146** — not significant

We do not reject the null. In the `main` runs the offsets span −298 to +365 steps against
plateaus of 3,675–8,525, i.e. below 10% of the plateau, and at or below the ≈152-step
checkpoint spacing. The detector **replaces** held-out labels after calibration; it does
not anticipate grokking.

### 4. Consistency with the Fourier circuit

A spectral analysis of the digit embeddings provides an independent check:

| group               | spectral concentration (top 6 of 48) |
| ------------------- | ------------------------------------ |
| generalising (n=12) | 0.952 ± 0.013                        |
| controls (n=4)      | 0.238 ± 0.073                        |
| white-noise level   | 6/48 = 0.125                         |

Counting key frequencies `k` (energy above 10× the median), the inequality **r\* ≥ 2k**
— expected since each frequency needs a sine and a cosine component — holds in **12/12
runs**, mean margin +3.42. Two independently constructed measures, one functional and one
spectral, agree.

Pearson correlation between 2k and r\* is +0.44 (p = 0.15, n = 12): weak, as expected,
since 2k is a lower bound rather than a point prediction.

**No frequency is shared by all runs** (27 distinct frequencies across the 12), so r*
recognises the dimensionality of the computed circuit rather than one particular solution.
Spectral concentration separates just as cleanly and needs no data at all — but only
someone who already knows the solution is Fourier can construct it. r* requires no such
knowledge.

---

## Setup

One-layer transformer on `(a + b) mod 97`:

- `d_model = 128`, 4 heads, `d_mlp = 512`, 221,952 parameters
- **no LayerNorm, no dropout, no biases** — the absence of LayerNorm keeps the embedding
  matrix directly interpretable in the Fourier basis
- full-batch AdamW, `lr = 1e-3`, `β = (0.9, 0.98)`, weight decay 1.0 (0.0 in `nc_wd0`)
- 30,000 steps (40,000 for `long`)

The four conditions were fixed by a scan over the training fraction, which recovers three
of the four regimes described by Liu et al. (2022):

| fraction | examples | t_mem | t_grok | regime        |
| -------- | -------- | ----- | ------ | ------------- |
| 0.20     | 1881     | 125   | never  | memorisation  |
| 0.25     | 2352     | 125   | 17825  | grokking      |
| 0.30     | 2822     | 125   | 6775   | grokking      |
| 0.35     | 3293     | 150   | 3875   | grokking      |
| 0.40     | 3763     | 150   | 1175   | comprehension |

0.30 was chosen over 0.25 because it is further from the memorisation boundary at 0.20,
where the time to grokking has high variance across seeds. 0.25 was kept as the
out-of-condition validation set.

---

## Repository structure

```
src/grokzip/
    data.py           modular-arithmetic dataset, deterministic split
    model.py          one-layer transformer
    train.py          full-batch loop, Config, grokking regime diagnostics
    checkpoints.py    hybrid log/linear grid, float32 checkpoint saving
    compression.py    the three operators: quantize, prune, truncate_rank
    measure.py        Evaluator: accuracy of a state_dict on train and test
    threshold.py      minimal viable rank r*, monotonicity check
    anchors.py        automatic selection of comparison checkpoints
    sweep.py          compression grids and systematic evaluation
    detector.py       R_crit calibration, exact sign test
    fourier.py        DFT of embeddings, key-frequency counting
    figdata.py        data aggregation for figures
    style.py          shared figure style

scripts/              one executable per phase
results/              all intermediate CSVs
figures/              report figures
report/               LaTeX source of the two-page report
```
