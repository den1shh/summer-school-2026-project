# SMILES-2026 Hallucination Detection — Solution Report

## Headline metrics

| Metric | Value |
|---|---:|
| Avg test accuracy (5-fold StratifiedKFold CV) | 78.22 % |
| Avg test F1 | 85.60 % |
| Avg test AUROC | 79.16 % |
| Avg train accuracy | 81.20 % |
| Avg train AUROC | 88.35 % |
| Majority-class baseline accuracy | 70.10 % |

Per-fold breakdown (5-fold StratifiedKFold, `random_state=42`):

| Fold | n_train | Test acc | Test F1 | Test AUROC |
|------|--------:|---------:|--------:|-----------:|
| 1 | 551 | 75.36 % | 83.33 % | 80.61 % |
| 2 | 551 | 79.71 % | 86.92 % | 80.44 % |
| 3 | 551 | 80.43 % | 87.08 % | 81.69 % |
| 4 | 551 | 81.16 % | 87.25 % | 77.43 % |
| 5 | 552 | 74.45 % | 83.41 % | 75.61 % |

---

## Reproducibility instructions

```bash
git clone https://github.com/den1shh/summer-school-2026-project.git
cd summer-school-2026-project
pip install -r requirements.txt
python solution.py
```

Outputs:

- `results.json` — per-fold and averaged metrics over the 5 internal folds.
- `predictions.csv` — `id,label` for the 100 unlabelled samples in
  `data/test.csv`.

### Environment

Reference timings on Kaggle GPU T4 x2: full pipeline in $\approx$ 5 min in total (160 s extraction + 2 min probe).

## Important implementation details

The first import of `aggregation.py` triggers a module-init forward
pass with `attn_implementation="eager"` over all 689 + 100 = 789
prompt + response sequences. The resulting 7840-dimensional feature matrix
(672-d for Lookback Lens + 7168-d for multilayer pools) is cached to
`.aggregation_cache_v11.npy`. Subsequent runs read from
the cache.

Eager attention is required because the default `flash-attention` /
`sdpa` backends do not materialise per-head attention weights, which
are needed for the Lookback Lens features.

## Final solution description

Two parallel logistic-regression classifiers, each receiving a different family of features, calibrated independently, then combined by a Brier-optimal soft-vote with a threshold tuned for accuracy on out-of-fold predictions.
```
                             prompt + response
                                    │
                                    ▼
              ┌────────────────────────────────────────────┐
              │                Qwen2.5-0.5B                │
              └────────────────────────────────────────────┘
                       │                         │
                       ▼                         ▼
               Attention tensors           Hidden states
                       │                         │
                       ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │  Lookback Lens   │      │  Multilayer pool │
              └──────────────────┘      └──────────────────┘
                       │                         │
                       │                         ▼
                       │                ┌──────────────────┐
                       │                │       PCA        │
                       │                └──────────────────┘
                       │                         │
                       ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │  StandardScaler  │      │  StandardScaler  │
              └──────────────────┘      └──────────────────┘
                       │                         │
                       ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │  Calibrated LR   │      │   Calibrated LR  │
              │   (C = 0.003)    │      │    (C = 0.01)    │
              └──────────────────┘      └──────────────────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         ┌────────────────────┐
                         │     Soft-vote      │
                         └────────────────────┘
                                    │
                                    ▼
                             label ∈ {0, 1}
```
### Feature pipeline

Because `evaluate.py` passes only `(hidden_states, attention_mask)` to
the aggregator at sample-time, all features are pre-computed at
module load. The pre-computation runs one extra forward pass of
Qwen 2.5-0.5B over the concatenated train + test corpus (789 samples)
with `attn_implementation="eager"`, `output_hidden_states=True`, and
`output_attentions=True`. At module load, all prompts are tokenised once and their lengths cached. A global counter then advances on each call to `aggregation_and_feature_extraction`, returning the matching pre-computed row.

**Prompt / response boundary detection.** For each `(prompt + response)`
sample we record the prompt token count via
`len(tokenizer(prompt, add_special_tokens=False)["input_ids"])` and
recover the response token span as `[prompt_n : seq_len]` inside the
concatenated tokenisation. The `add_special_tokens=False` flag is
critical: with the default (`True`) Qwen's tokeniser would prepend
extra BOS-style tokens that do not appear in the joint
`tokenizer(prompt + response)`, shifting the boundary by a fixed
offset and silently corrupting the Lookback Lens prompt-vs-response
attention partition.

**Lookback Lens features (672-d).**
For each of layers L11 – L16, each of 14
query heads (Qwen 2.5-0.5B has 14 Q and 2 KV heads per layer), and over
the response token span, we compute 8 statistics per
(layer, head):

| # | Statistic | Definition |
|---|---|---|
| 0 | `lookback_mean` | mean over response tokens of `A_prompt / (A_prompt + A_resp)` — the Lookback Lens ratio of Chuang et al. [1] |
| 1 | `lookback_min` | minimum lookback ratio across response tokens (worst single attention shift) |
| 2 | `lookback_max` | maximum lookback ratio across response tokens |
| 3 | `attn_entropy_mean` | mean Shannon entropy of the row-normalised attention vector over real tokens |
| 4 | `attn_entropy_min` | sharpest attention spike (faithful answers tend to have sharper alignments) |
| 5 | `attn_entropy_max` | most diffuse attention |
| 6 | `attn_to_sink` | mean fraction of attention going to the very first token (BOS-style activation sink, Sun et al. [8]) |
| 7 | `attn_to_resp` | mean fraction of attention going to the response itself |

Total dimensions: 6 layers $\times$ 14 heads $\times$ 8 stats = 672.

Lookback ratio is the signal proposed by
Chuang et al. [1]: faithful answers attend to prompt,
hallucinations shift to the response. mean/min/max give length-invariant
summaries. Entropy stats separate sharp retrieval from diffuse
confabulation (Voita et al. [9]). `attn_to_sink` reflects
BOS-style activation sinks (Sun et al. [8]). L11–L16 are the
middle of Qwen's 24 layers — prior probes identify them as carrying
truthfulness signal (Azaria & Mitchell [4]; Li et al. [10]).

**Multilayer hidden-state features (7168-d -> 128-d (PCA)).**
For each of layers L13, L17, L21, L24 (a roughly geometric layer
spacing chosen to span mid-to-final depth), we extract two pools over
the response token span:
- the last response token representation
- the mean of response token representations.

Concatenating last + mean across 4 layers yields 8 $\times$ 896 = 7168,
which PCA inside `probe.py` reduces to 128-d before the LR
classifier sees it. PCA is fit on the entire training set.

Last-token pooling captures the most context-saturated representation
— every response token has attended to all preceding tokens, including
the full prompt — while mean pooling is length-invariant and
noise-robust. The two are complementary statistics that together
preserve distinct components of the response trajectory.

### Probe (`probe.py`)

Two parallel logistic-regression classifiers, each receiving a
different feature slice, calibrated independently, then combined by a
single Brier-optimal soft-vote.

**Per-base out-of-fold probabilities.** Inside `fit()` we run a
5-fold $\times$ 5-repeats = 25-fold stratified OOF protocol on the training
data the outer fold sees. Each fold trains a
`CalibratedClassifierCV(LogisticRegression, method="isotonic", cv=3)`
on the in-fold training subset and predicts on the held-out subset,
producing one calibrated probability per training sample per base.
This gives the two 689-vector OOF probability streams used for weight
optimisation and threshold tuning. All calibrated classifiers from
the 25 inner folds are retained and averaged at predict time.

**Brier-optimal weights instead of uniform soft-vote.** With two
bases of different signal strength, uniform averaging under-weights
the stronger one. We solve the constrained least-squares problem

$$\min_{w \succeq 0,\; \mathbf{1}^\top w = 1} \frac{1}{n}\sum_{i=1}^n \big(\langle w, p_i\rangle - y_i\big)^2$$

over the simplex via SciPy's `Nelder-Mead`. On the OOF probabilities this consistently
converges to `w ≈ (lookback ≈ 0.68, multilayer ≈ 0.32)`, recovering
the relative signal strength of the two bases.

**Threshold tuning on OOF for accuracy.** The decision threshold
is tuned to maximise accuracy on the OOF blend. Empirically this gives `thr ≈ 0.49`, very close to 0.5 — confirming
that the soft-vote is well-calibrated.

The decisive design question was: which combination of feature
families (attention patterns, hidden state geometry, token-level
uncertainty, topological / spectral structure) is actually informative? 

We conducted a correlation analysis of out-of-fold probability streams. 
During the methods sweep we computed 5 distinct probe bases that the
literature points to as candidate hallucination signals for
single-pass internal-state probing of a causal LM — `lookback`
(Chuang et al. [1]), `toha` (Bazarova et al. [2]),
`llmcheck` (Sriramanan et al. [5]), `spectral`
(Binkowski et al. [6]), and `multilayer` (Azaria & Mitchell [4];
Marks & Tegmark [3]) — and we measured
the 5 $\times$ 5 Pearson correlation matrix of their OOF probability
outputs (each base trained with the same calibrated LR head). The
three bases other than `lookback` and `multilayer` were retained
only for this sweep — the submitted `aggregation.py` no longer
computes any of them.
Correlation of every base with `lookback` (the strongest single
feature):

| Base | Family | Standalone AUROC | cor(·, lookback) |
|---|---|---:|---:|
| `lookback` | Attention pattern (per-token prompt vs response ratio) | 79.06 % | 1.000 |
| `toha` | Topological divergence on attention graph (MTop-Div) | 77.95 % | 0.906 |
| `llmcheck` | LogDet of hidden + attention covariances | 74.46 % | 0.817 |
| `spectral` | Laplacian eigenvalues of response-region attention | 73.75 % | 0.804 |
| `multilayer` | Hidden state pooling (last + mean per layer) | 76.67 % | 0.704 |

Every attention-derived base — topological divergence, LogDet,
Laplacian eigenvalues — has correlation 0.80 – 0.91 with
Lookback Lens. They are near-copies of the same probability stream,
despite ostensibly measuring very different mathematical properties.

`multilayer` has the lowest correlation with `lookback` (0.70) — its
information lives in the hidden-state geometry, not the attention
pattern, which makes it the natural complement to Lookback in a
two-base ensemble.

Empirical study shows that the 2-base architecture
(`lookback` + `multilayer`) matches every richer ensemble we tried. 

### What contributed most to the metric

In rough order of contribution, relative to the unmodified baseline
last-token MLP (test accuracy 71.00 %):

| Change | Cumulative test acc | Δ |
|---|---:|---:|
| Baseline last-token MLP                                              | 71.00 % | — |
| Replace MLP with L2-regularized logistic regression                                 | 71.7 %  | + 0.7 |
| Multi-layer pooling instead of single layer                          | 72.6 %  | + 0.9 |
| Add Lookback Lens attention features                             | 75.5 %  | + 2.9 |
| Combine Lookback + multilayer via Brier-optimal soft-vote        | 78.0 %  | + 2.5 |
| Isotonic per-base calibration (replacing raw LR probabilities)       | 78.4 %  | + 0.4 |
| 5 $\times$ 5 OOF protocol for stable weight estimation                    | 78.7 %  | + 0.3 |

The two largest jumps are adding Lookback Lens (+ 2.9) and
combining it with multilayer via Brier-optimal weights (+ 2.5).

## Failed attempts

### Pseudo-labelling

Inspired by self-training pipelines, we ran three rounds of self-training: take the most-confident
test predictions, label them with the current
model's prediction, fold into training, retrain. The OOF AUROC change was
zero to slightly negative:

### LLM-as-judge

Implemented a few-shot judge prompt: given context, question, and
candidate response, ask Qwen 2.5-0.5B to answer "SUPPORTED" or
"NOT_SUPPORTED" and read the logit ratio of the two completion tokens.
Standalone OOF AUROC: 59.46 % — effectively random.

### Trajectory / cross-layer drift features

Per-layer hidden-state L2 norm, layer-to-layer L2 drift,
layer-to-layer cosine similarity, first-to-last hidden state shift —
concatenated. In the ensemble:
zero Brier weight.

### Per-head ITI-style selection

Inference-Time Intervention's top-K head probing (Li et al. [10])
was implemented: train a probe on each individual attention
head's residual contribution, select the top-K by per-head AUROC,
ensemble. Underperformed both
Lookback and multilayer on absolute terms and strongly correlated with Lookback. 

### HaloScope-style unlabelled SVD subspace

Du et al. [7] propose HaloScope: extract hidden states at
a "best" layer (we tried L20), centre, run TruncatedSVD(32), use
the projection as features. The method was strictly worse than supervised
multilayer pooling at this dataset scale.


## References

1. **Chuang, Y.-S., Qiu, L., Hsieh, C.-Y., Krishna, R., Kim, Y., &
   Glass, J. (2024).** *Lookback Lens: Detecting and Mitigating
   Contextual Hallucinations in Large Language Models Using Only
   Attention Maps*. EMNLP 2024. arXiv:2407.07071.

2. **Bazarova, A. et al. (2025).** *Hallucination Detection in LLMs
   with Topological Divergence on Attention Graphs*.
   arXiv:2504.10063.

3. **Marks, S. & Tegmark, M. (2023).** *The Geometry of Truth:
   Emergent Linear Structure in LLM Representations of True / False
   Datasets*. arXiv:2310.06824.

4. **Azaria, A. & Mitchell, T. (2023).** *The Internal State of an
   LLM Knows When It's Lying* (SAPLMA). EMNLP-Findings 2023.
   arXiv:2304.13734.

5. **Sriramanan, G. et al. (2024).** *LLM-Check: Investigating
   Detection of Hallucinations in Large Language Models*.
   NeurIPS 2024.

6. **Binkowski, J. et al. (2025).** *Hallucination Detection in
   LLMs Using Spectral Features of Attention Maps*. arXiv:2502.17598
   (LapEigvals).

7. **Du, X., Xiao, C., & Li, S. (2024).** *HaloScope: Harnessing
   Unlabeled LLM Generations for Hallucination Detection*.
   NeurIPS 2024.

8. **Sun, M. et al. (2024).** *Massive Activations in Large
   Language Models*. ICML 2024. arXiv:2402.17762.

9. **Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I.
   (2019).** *Analyzing Multi-Head Self-Attention: Specialized
   Heads Do the Heavy Lifting, the Rest Can Be Pruned*. ACL 2019.
   arXiv:1905.09418.

10. **Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M.
    (2023).** *Inference-Time Intervention: Eliciting Truthful
    Answers from a Language Model*. NeurIPS 2023. arXiv:2306.03341.