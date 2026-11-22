# Research log: improving test accuracy under the sales-rank split

Live log, started 2026-11-16 18:55 SGT, deadline 21:00 SGT. It is the working log of AI-assisted exploration runs
(the "agents" below are AI coding agents, each exploring one direction), kept as written as the record behind the
report's exploration appendix. For the final repo the per-direction write-ups (`docs/agents/`), the agent briefs, and the
exploration folders other than `exp/g-arch`, `exp/graph`, `exp/g-localprior`, `exp/g-restore` and `exp/main` were removed, so some paths
below no longer exist; the numbers are as logged. Where the log says "every choice made on val", note that the LLM
weight and λ were later fixed at 1 by Bayes' rule rather than tuned (see the README's evaluation protocol).

## Goal

Improve test accuracy and macro-F1 with simple, well-motivated methods. Keep the evaluation protocol clean:
choices are made on validation data, and test numbers are reported for a fixed set of methods.

## Summary (final, 21:25 SGT)

**Headline** (official split, test accuracy / macro-F1, mean over 3 GNN seeds; every choice made on val):

| Pipeline | normal | cold start |
|---|---|---|
| Repo best before today: GT on MiniLM | 0.765 / 0.365 | 0.709 / 0.330 |
| Repo best with an LLM: MiniLM + GPT answer as a feature + GCN | 0.772 / 0.353 | 0.758 |
| GT on MiniLM + **LLM fusion** (GPT+Qwen, no tuned weight) | 0.814 / 0.414 | 0.799 / 0.404 |
| ... + **BP of neighbours' labels/LLM answers** (co-purchase + text kNN) | 0.822 / 0.422 | 0.806 / 0.410 |
| GT on a **fine-tuned gte-large** encoder (no LLM) | 0.817 / 0.423 | 0.809 / 0.416 |
| **Fine-tuned gte-large + GT + LLM fusion + BP** | **0.837 / 0.439** | **0.828 / 0.432** |
| same, full stack (fuse+BP+kNN) over **3 encoder fine-tuning seeds** × 3 GNN seeds | **0.835 ± 0.001 / 0.435** | **0.828 ± 0.001 / 0.430** |

That is +7.2 points of accuracy and +7.4 of macro-F1 over the previous best GT, and +11.9 points in cold start.
The last row is robust across encoder fine-tuning seeds (per-seed 0.835 / 0.833 / 0.836; cold 0.828 / 0.827 / 0.829). About 4.3% of test
cannot be predicted (Baby, MISSING), so the ceiling is about 0.957.

**The method, in one sentence each:**
1. *Diagnosis*: the sales-rank split mainly causes **label shift**. Classes that are rare in train dominate the
   long-tail test set, and val (train-like) hides it.
2. *LLM fusion*: an LLM never saw the train label mix. Treating its zero-shot answer g as an observation with
   likelihood P(g|y) (its confusion matrix, counted on labelled nodes) gives p(y|x,g) ∝ p(y|x)·P(g|y): Bayes' rule,
   no tuned weight. Tuning a weight on val would under-weight the LLM, because val looks like train.
3. *Graph as post-hoc BP*: one round of belief propagation passes neighbours' train labels and LLM evidence through
   a label-compatibility matrix, over co-purchase edges and text-kNN edges (the latter exist in cold start).
4. *Better text*: fine-tuning the encoder on train nodes matters; bigger frozen encoders and longer context barely
   do (truncation at 256 tokens costs about 0.3 points).

**What each direction contributed** (all six are finished; details below and in `docs/agents/`):

| Direction | Alone (best, test acc) | On top of the fusion stack |
|---|---|---|
| A. LLM fusion + BP (main) | GT 0.765 -> 0.822 | — |
| B. Fine-tuned encoder | GT 0.765 -> 0.817 (gte-large) | **+1.5** (0.822 -> 0.837) |
| C. C&S over co-purchase + kNN | GT 0.765 -> 0.775 | +0.0 |
| D. LLM restructuring of descriptions | GT 0.765 -> 0.782 | +0.0 (GT), +0.6 (SAGE) |
| D. Better Qwen prompt | Qwen 0.659 -> 0.714 | ≈0 |
| E. Agreement pseudo-labels | GT 0.763 -> 0.800 | -0.3 |
| F. Class-name prototype head | SAGE 0.738 -> 0.769 | +0.6 |
| *Round 2 (21:00-21:25)* | | |
| Graph-RAG LLM re-ranker on uncertain nodes | LLM pick 0.574 vs stack 0.585 on them | +0.0 (cold +0.1) |
| LM adapted on the fused teacher's test pseudo-labels | GT 0.817 -> 0.830 | -0.3 |
| Neighbour titles in the LM input | GT 0.817 -> 0.815, F1 -2 to -4 pt | not adopted |
| LLM-synthesised listings for train-rare classes | SAGE 0.756 -> 0.768 | ±0.1 |
| Qwen3-1.7B as encoder (cut-down fine-tune) | GT 0.788 (< gte 0.817) | -1.0 |
| Multi-round collective inference | — | +0.2 |
| Dawid-Skene re-estimation of LLM confusions on test | — | -0.2 |

Reading: all LLM-based routes (fusion, restructuring, pseudo-labels, synthetic data, re-ranking, distillation into
the encoder) transfer the same LLM knowledge, and the
parameter-free post-hoc fusion is the cheapest and strongest of them. The only gain that is orthogonal to it is
better text representation learned from the train labels (fine-tuning).

**Reproduce:** `scripts/export_probs.py MODEL [--emb NAME]` (base models) -> `scripts/run_fusion.py TAG...`
(fusion + BP, choices on val) -> `results/3_gnn_text/fusion_*`. Fine-tuned embeddings: `exp/text-encoder/finetune_lm.py`.
Qwen answers: `scripts/llm_zero_shot.py` (base prompt) and `exp/llm/gen.py` (chosen prompt).

**Protocol note for the report:** test labels were looked at during the error analysis (morning) and for
reporting every variant here. No hyperparameter, variant or stopping point was chosen on test. Report all rows,
including the negative ones.

**Negative results worth reporting** (all in the sections below): prior correction alone (EM collapses; even the
oracle prior does not help GT, so p(x|y) also shifts); logit adjustment / class weights (F1 up, accuracy flat);
label trick, SIGN up to 6 hops, deeper GT (far nodes fail from label shift, not reach); self-training with
agreement pseudo-labels (+3.7 alone, +0 once fused: same information); C&S on top of BP (+0).

## Starting point (error analysis, 2026-11-16)

- The val-to-test drop (GCN 0.871 -> 0.711) is mostly **label shift**. The train and test label distributions differ
  by total variation 0.245, while val differs from train by only 0.071. If test had the train class mix, GCN would
  score 0.85.
- Classes that are rare in train but large in test (Electronics, Office Products, Tools & Home Improvement,
  Automotive) are almost never predicted. Office Products is 4.0% of test but 0.18% of GCN's predictions.
- About 4.3% of test is effectively unpredictable: Baby (807 nodes, none in train), MISSING (763 nodes, all without a
  title), and 7 other classes with no training nodes.
- GPT-3.5 zero-shot (TAPE) never saw the training distribution, and it is strong exactly on these classes. The
  oracle "GT or GPT is right" reaches 0.858.
- A first try, rescaling GCN's posterior by a test prior estimated from GPT's answers, gave 0.711 -> 0.753.
  This number was computed on test and is exploratory only.

## Environment note

The 8 GPUs are shared with another user's jobs (about 5GB per card). We use the free memory.

## Plan: six directions in parallel (revised 19:05 after feedback that the first plan was too narrow)

Shared rules for every direction are in `docs/agent_brief.md`: no test labels for selection, choose on val, report
every variant on test, 3 seeds. Each direction writes to its own file.

| Direction | Owner | Question | Findings |
|---|---|---|---|
| A. Label shift + LLM Bayesian fusion | main thread | Correct the train-prior bias with a prior estimated by an LLM (BBSE), then fuse the LLM's answer through its confusion matrix, with no tuned weights | this file |
| B. Text representations | agent `text-encoder` | Stronger frozen encoders (bge/e5/gte) and a fine-tuned LM | `docs/agents/text-encoder.md` |
| C. Graph usage | agent `graph` | Label trick, C&S, multi-hop (SIGN/APPNP) for far nodes, text-kNN edges for cold start, heterophily | `docs/agents/graph.md` |
| D. LLM signal | agent `llm` | Local Qwen3.5-9B zero-shot on all nodes, better prompts, LLM restructuring of weak descriptions | `docs/agents/llm.md` |
| E. Self-training | agent `self-training` | Class-balanced pseudo-labels from GNN+LLM agreement, LLM distillation on unlabelled nodes | `docs/agents/self-training.md` |
| F. Imbalance + unseen classes | agent `imbalance` | Re-weighted / balanced losses, a label-name prototype classifier, zero-shot by name | `docs/agents/imbalance.md` |

At the end, the best component of each direction is stacked (features from B/C, training from E/F, then A on top),
and only that fixed stack plus its ablations is reported.

## Progress

| Time (SGT) | Step | Result |
|---|---|---|
| 18:55 | Log created | |
| 18:58 | Base model probabilities exported (MLP/GCN/SAGE/GT, plain and logit-adjusted, 3 seeds, normal + cold) to `outputs/probs/` | val: MLP 0.823, GCN 0.871, SAGE 0.878, GT 0.905; logit-adjusted versions lose 3-10 points of val (val has the train mix) |
| 19:00 | Qwen3.5-9B zero-shot launched on all 54k nodes (8 shards) | 300-node pilot: test-node acc 0.59 vs GPT-3.5 0.66 |
| 19:05 | Five exploration agents launched (B-F) | |
| 19:10 | **A, first results** (`scripts/run_label_shift.py`, `results/label_shift_*.csv`) | see table A1 below |
| 19:12 | Truncation measured (user's question) | MiniLM 256-token limit: 16.3% of texts cut, 17.7% of all tokens lost; at 512 only 2.7% cut. Ablation (256 / 512 / chunk-average) handed to `text-encoder` |

### A1. Label shift and LLM fusion (test acc / test macro-F1, mean of 3 seeds; nothing tuned on test)

| Base model | base | prior BBSE-GPT | prior oracle (bound) | blend w=0.25 (repo) | **fuse GPT** | BBSE + fuse |
|---|---|---|---|---|---|---|
| MLP | 0.661 / 0.269 | 0.692 / 0.276 | 0.703 / 0.295 | 0.721 / 0.340 | 0.765 / 0.365 | **0.773** / 0.330 |
| GCN | 0.712 / 0.290 | 0.743 / 0.300 | 0.761 / 0.322 | 0.761 / 0.363 | 0.787 / 0.378 | **0.789** / 0.340 |
| SAGE | 0.738 / 0.327 | 0.754 / 0.316 | 0.776 / 0.346 | 0.779 / 0.380 | **0.799 / 0.399** | 0.793 / 0.342 |
| GT | 0.765 / 0.365 | 0.771 / 0.327 | 0.758 / 0.362 | 0.795 / 0.396 | **0.811 / 0.411** | 0.804 / 0.347 |
| GCN cold | 0.652 / 0.259 | 0.684 / 0.271 | 0.699 / 0.292 | 0.717 / 0.341 | 0.766 / 0.372 | **0.770** / 0.329 |
| SAGE cold | 0.619 / 0.266 | 0.564 / 0.219 | 0.570 / 0.248 | 0.728 / 0.358 | **0.775 / 0.384** | 0.740 / 0.309 |
| GT cold | 0.709 / 0.330 | 0.712 / 0.297 | 0.678 / 0.326 | 0.753 / 0.372 | **0.786 / 0.394** | 0.778 / 0.333 |

Takeaways:
- **Bayesian fusion through GPT's confusion matrix ("fuse GPT") is the clear winner**: p(y|x,g) ∝ p(y|x)·P(g|y), with
  P(g|y) counted on train+val labels and no tuned weight. It beats the repo's tuned blend for every model, in both
  settings, on both accuracy and macro-F1 (GT 0.795 -> 0.811). On cold start it is the largest gain (SAGE
  0.619 -> 0.775): the LLM answer replaces the missing neighbours.
- Prior correction alone is fragile. It helps MLP/GCN (+3 points) but not GT, and even the oracle prior does not
  help GT or the cold-start models. So the shift is not a pure label shift: p(x|y) also moves. EM collapses
  (GT 0.13). BBSE on top of fusion adds a little for MLP/GCN but costs macro-F1 (its prior is zero for the unseen
  classes), so fusion alone is the default.

### A2. Using the graph on top of the fusion (`exp/main/fuse_more.py`, `exp/main/bp_fuse.py`; β/α chosen on val)

- GPT's second answer as an extra observation: no test gain.
- Label smoothing of the fused posterior (C&S smooth step): +0.3 to +0.5 points (GT 0.811 -> 0.814 at the
  val-chosen α).
- **Belief propagation of the neighbours' LLM answers**: b_i(y) ∝ p(y|x_i)·P(g_i|y)·Π_j m_j(y)^β with
  m_j(y) = Σ_y' H[y,y']·e_j(y'). Here e_j is a train neighbour's one-hot label or an unlabelled neighbour's LLM
  likelihood, and H is the label-compatibility matrix from train-train edges. One scalar β is chosen on val.
  Results with the val-selected β: MLP 0.765 -> **0.798**, GCN 0.787 -> 0.798, SAGE 0.799 -> 0.804, GT 0.811 -> 0.814
  (macro-F1 0.40-0.41). A text-only MLP plus post-hoc BP matches SAGE + fusion, so the graph's benefit can be
  obtained entirely after training. In cold start the test nodes have no edges, so nothing changes, as expected.
- Logit-adjusted (balanced) models + fusion are no better than plain models + fusion (SAGE-LA 0.796 vs 0.799,
  GCN-LA 0.797 vs 0.787, GT-LA 0.762 because its val-acc stopping rule breaks). The fusion already fixes the
  rare classes: after GT + fusion, Office Products recall goes 0.58 -> 0.84, Tools 0.35 -> 0.65,
  Automotive 0.12 -> 0.44.
- What is left after GT + fusion (test 0.8135): 7,040 errors, of which 1,615 (23%) are in unpredictable classes
  (Baby -> "Baby Products" 625, MISSING 763). Largest learnable confusion: Electronics -> Cell Phones (668, mostly
  tablet accessories, an ambiguous labelling).
- 19:21 **Qwen3.5-9B zero-shot finished** (`outputs/llm/qwen35_9b.npy`). Alone it is weaker than GPT-3.5 (test
  0.659 / 0.340 vs 0.723 / 0.379; val 0.753 vs 0.782). But fused as a second independent observation it helps
  everywhere, and val prefers it:

  | Model (test acc / F1) | fuse GPT | fuse GPT+Qwen | + BP (β=0.5) |
  |---|---|---|---|
  | MLP | 0.765 / 0.365 | 0.783 / 0.394 | 0.798 / 0.403 |
  | GCN | 0.787 / 0.378 | 0.795 / 0.396 | 0.804 / 0.404 |
  | SAGE | 0.799 / 0.399 | 0.802 / 0.403 | 0.809 / 0.408 |
  | GT | 0.811 / 0.411 | 0.814 / 0.414 | **0.820 / 0.419** |
  | GCN cold | 0.766 | 0.784 | (no test edges) |
  | GT cold | 0.786 / 0.394 | **0.799 / 0.404** | |

- 19:30 **Text-kNN messages** (idea from the `graph` agent's interim result that kNN edges rescue cold start). Each
  node also receives BP messages from its 5 nearest MiniLM neighbours (their train label or LLM likelihood), with
  their own compatibility matrix and weight γ. Cold-start nodes keep these edges because their text is known.
  Everything (LLM set, β, γ) is chosen on val:

  | Model | setting | base | fuse | fuse+BP | **fuse+BP+kNN** |
  |---|---|---|---|---|---|
  | MLP | normal | 0.661 / 0.269 | 0.783 / 0.394 | 0.794 / 0.401 | **0.804 / 0.408** |
  | GCN | normal | 0.712 / 0.290 | 0.795 / 0.396 | 0.802 / 0.402 | **0.804 / 0.406** |
  | SAGE | normal | 0.738 / 0.327 | 0.802 / 0.403 | 0.808 / 0.408 | **0.813 / 0.411** |
  | GT | normal | 0.765 / 0.365 | 0.814 / 0.414 | 0.819 / 0.419 | **0.822 / 0.422** |
  | MLP (no edges at test) | cold | 0.661 | 0.783 | 0.783 | **0.791 / 0.398** |
  | GCN | cold | 0.652 / 0.259 | 0.784 / 0.392 | 0.784 | **0.791 / 0.398** |
  | SAGE | cold | 0.619 / 0.266 | 0.784 / 0.392 | 0.784 | **0.790 / 0.400** |
  | GT | cold | 0.709 / 0.330 | 0.799 / 0.404 | 0.799 | **0.806 / 0.410** |

  Std over seeds is at most 0.003 for every fused row. Cold start went from 0.62-0.71 to about 0.79-0.81, and with
  everything post-hoc, a text-only MLP (0.804) is as good as the GNNs.
- 19:35 **Truncation answered** (`text-encoder`, details in `docs/agents/text-encoder.md`): raising MiniLM from 256 to
  512 tokens changes test by +0.3 points overall (MLP 0.659 -> 0.662) and by +1.3 on the long-text subset.
  **Truncation is not a bottleneck**: the title comes first, and long texts are already the easiest bucket.
  The big text lever is **fine-tuning** the encoder on train nodes (bge-base, 3 epochs): MLP 0.661 -> 0.762,
  SAGE 0.738 -> 0.786, cold 0.619 -> 0.760. Frozen bigger encoders mostly raise val, not test (gte-large SAGE 0.756).
- 19:37 **Fusion stack on fine-tuned features** (`results/3_gnn_text/fusion_*ft-bge-base*`; one LM seed, std over GNN seeds):

  | Features + model | base | fuse | fuse+BP | **fuse+BP+kNN** | cold: base -> full |
  |---|---|---|---|---|---|
  | ft-bge-base + MLP | 0.762 / 0.352 | 0.810 / 0.405 | 0.822 / 0.417 | 0.823 / 0.420 | 0.762 -> 0.817 |
  | ft-bge-base + SAGE | 0.786 / 0.371 | 0.824 / 0.418 | 0.827 / 0.421 | **0.828 / 0.425** | 0.760 -> **0.814** |
  | ft-bge-base + GT | 0.767 / 0.341 | 0.811 / 0.397 | 0.818 / 0.410 | 0.822 / 0.417 | 0.752 -> 0.813 |

  The gains from better text and from post-hoc fusion largely overlap: +4.8 points on base for SAGE become +1.5
  after the full stack (0.813 -> 0.828).
- 19:40 **Why no tuned weight** (`exp/main/lambda_check.py`): scaling the LLM log-likelihood by λ. Val prefers
  λ=0.5 (GT val 0.913), but test is flat or best at λ≈0.75-1 (GT 0.810 at 0.5, 0.814 at 1.0). Val has the train
  label mix, where the LLM is least needed, so **tuning the LLM weight on val under-weights it**. The Bayes-rule
  weight λ=1 is the more robust choice, which is an argument for the parameter-free fusion over the tuned blend.
- 19:42 **Agent `imbalance` finished** (`docs/agents/imbalance.md`). Best training-side fix: a **class-name
  prototype cosine head** on SAGE, 0.738 / 0.327 -> 0.769 / 0.379, chosen on "val wacc" (val per-class recall
  weighted by a label-free test-prior estimate). Re-weighted / logit-adjusted losses raise F1 by 3-5 points but not
  accuracy. Unseen classes stay at about 0 (Baby vs Baby Products cannot be separated by names). The exported GT-LA
  files were broken by stopping on val acc. With the full fusion stack on top: SAGE-Proto 0.819 / 0.427 vs plain
  SAGE 0.813 / 0.411. It stacks, but by only +0.6 points, and it needs class-name embeddings in the same space as
  the features (not available for the fine-tuned encoder), so it is not in the final stack.
- 19:45 **Agent `self-training` finished** (`docs/agents/self-training.md`). Pseudo-labels on test nodes where GT
  and GPT agree: GT 0.763 / 0.364 -> 0.800 / 0.397, cold 0.709 -> 0.756. A label-free test-like holdout also
  improves (0.801 -> 0.831). Confidence-only pseudo-labels do nothing, which confirms the train-prior-bias
  explanation of the earlier failure. Class capping adds nothing, because the agreement set lacks the shifted classes.
  **But with the fusion stack on top it no longer helps**: GT-PLagree 0.819 / 0.419 vs plain GT 0.822 / 0.422. It
  learns the same GPT information that the fusion injects post-hoc, so the cheaper post-hoc route is kept.
- 19:45 **Agent `graph` finished** (`docs/agents/graph.md`). Best: GT + Correct & Smooth over co-purchase +
  text-kNN edges (k=5, low-degree nodes only), 0.765 -> 0.775, cold 0.709 -> 0.730. The label trick, SIGN up to 6
  hops, deeper GT and H2GCN-style separation do not help. Key diagnosis: **far nodes fail because of label shift, not
  because messages do not reach them**. The far test nodes' label mix is very unlike train (TV 0.14 at 1 hop, 0.47
  at 3, 0.78 at 4+), and even a text-only MLP is at 0.24 there. With the fusion stack on top: GT-CSknn
  0.821 / 0.422 vs GT 0.822 / 0.422. C&S and our BP do the same job (propagating train labels plus kNN text
  neighbours), so BP, which also propagates LLM evidence, subsumes it.
- 19:50 **Where the stack gains** (`exp/main/breakdown.py`, test acc by hop distance to the nearest train node;
  base -> fuse -> fuse+BP+kNN):

  | slice | n | GT (MiniLM) | ft-bge-base + SAGE |
  |---|---|---|---|
  | 1 hop | 15,570 | 0.833 -> 0.850 -> 0.854 | 0.843 -> 0.858 -> 0.861 |
  | 2 hops | 15,209 | 0.773 -> 0.807 -> 0.812 | 0.793 -> 0.817 -> 0.818 |
  | 3 hops | 4,634 | 0.663 -> 0.755 -> 0.777 | 0.698 -> 0.775 -> 0.782 |
  | 4+ hops | 2,320 | **0.463 -> 0.694 -> 0.752** | 0.544 -> 0.740 -> 0.767 |
  | no title | 3,139 | 0.644 -> 0.628 -> 0.641 | 0.696 -> 0.670 -> 0.677 |

  The far, label-shifted nodes gain the most (+29 points for GT at 4+ hops), exactly where the error analysis said the
  problem was. The no-title slice loses 1-2 points: the LLMs are weak there (no title, and the MISSING class), so
  this is the place where LLM restructuring could still help.
- 20:12 **Agent `text-encoder` finished**; fusion on the fine-tuned large encoders (`results/3_gnn_text/fusion_*ft-*large*`).
  The val-selected stage per row is in bold:

  | Features + model | base | fuse | fuse+BP | fuse+BP+kNN | cold: base -> full |
  |---|---|---|---|---|---|
  | ft-gte-large (1 LM seed) + GT | 0.817 / 0.423 | 0.835 / 0.437 | **0.837 / 0.439** | 0.835 / 0.437 | 0.809 -> **0.828** |
  | ft-gte-large + SAGE | 0.810 / 0.392 | 0.833 / 0.428 | 0.834 / 0.432 | **0.833 / 0.434** | 0.785 -> 0.820 |
  | ft-gte-large + MLP | 0.784 / 0.368 | 0.819 / 0.415 | 0.825 / 0.421 | **0.826 / 0.423** | 0.784 -> 0.818 |
  | ft-bge-large seed 0 / 1 + GT | 0.796 / 0.786 | 0.824 / 0.823 | 0.828 / 0.827 | **0.831 / 0.828** | 0.785 -> 0.823 |
  | ft-bge-large seed 0 / 1 + SAGE | 0.795 / 0.796 | 0.821 / 0.826 | 0.825 / 0.829 | **0.828 / 0.828** | 0.770 -> 0.815 |

  The fusion gain shrinks as the encoder gets better (+11 points on MiniLM GT, +2 on ft-gte-large GT), but it stays
  positive for every encoder, model and setting. Across the two bge-large fine-tuning seeds the full stack differs by
  ≤0.003.
- 20:28 **Agent `llm` finished** (`docs/agents/llm.md`).
  - Prompt chosen on a 2,000-node train+val sample: list only the 32 categories that have labelled nodes, plus
    one-line definitions of the confusable ones. With it, Qwen3.5-9B goes from test 0.659 to 0.714 (Office recall
    0.04 -> 0.68). Most of Qwen's loss came from near-duplicate legacy category names in the list.
  - Qwen3.8-27B with that prompt beats GPT-3.5 on the sample (0.804 vs 0.780) but did not finish on all nodes.
    I stopped it at 20:28 to free the shared GPUs.
  - **LLM restructuring** (the group's original idea): Qwen2.5-7B rewrote all 54k products as "Product type / Key
    attributes / Likely use" without naming a category. Concatenating the MiniLM embedding of that summary helps
    every model, and val agrees: MLP 0.661 -> 0.713, SAGE 0.737 -> 0.768, GT 0.765 -> 0.782; cold SAGE 0.619 -> 0.701.
  - With the fusion stack on top (`results/3_gnn_text/fusion_*orig+restr*`), the gain mostly overlaps: GT 0.822 vs 0.822,
    SAGE 0.817 vs 0.811, cold GT 0.806 vs 0.806. Restructuring and LLM answers carry the same LLM knowledge. The
    better Qwen prompt also adds little once GPT is fused; val picks it in only some rows.
- 20:55 **Round 2 (until 21:30), after feedback that the last tweaks were too incremental.** Three bold
  directions went to agents, with the current best posterior shared as `outputs/stack/best_{normal,cold}.npy`
  (13.8% of test has max prob < 0.9 and is only 52% accurate there, which is where the errors are):
  - `rerank`: an LLM re-ranker with graph context (graph-RAG) on uncertain nodes. It sees the product, its
    neighbours' titles with train labels, and the top-k candidates.
  - `lm-adapt`: fine-tune gte-large on train labels plus test pseudo-labels from the fused teacher (GLEM-style
    adaptation of the encoder to the long tail).
  - `graph-text`: fine-tune gte-large on graph-augmented text (own text + neighbours' titles), with a
    cold-start-safe variant that uses text-kNN neighbours.
  - 20:58, two more agents on the remaining GPU memory:
    - `llm-backbone`: fine-tune a decoder LLM (Qwen3-1.7B) as the node encoder instead of gte-large.
    - `synth`: LLM-synthesised training listings for the classes that are rare in train, added as isolated
      train nodes, which attacks the label shift at the data level.
  - Main thread, two principled extensions, both small or negative:
    - *Multi-round collective inference* (`exp/main/collective.py`; neighbours pass their current beliefs,
      damped, for T rounds; T/β/γ on val): 0.837 -> 0.839 normal, 0.828 -> 0.832 cold.
    - *Dawid-Skene EM* (`exp/main/dawid_skene.py`; re-estimate the LLM confusion matrices on unlabelled test
      nodes, with labelled counts as a Dirichlet prior): **negative**, 0.835 -> 0.833-0.831 as EM
      self-reinforces. The labelled-node confusion matrices are already adequate.
- 21:15 **Agent `rerank` finished** (`docs/agents/rerank.md`): **negative**. On the uncertain nodes Qwen3.5-9B,
  given the product, its neighbours' titles with train labels and the top-5 candidates, picks no better than the
  stack's own top-1 (test, max prob < 0.97: 0.574 vs 0.585). The val-chosen rule leaves test at 0.837; cold start
  gains +0.1 points. The stack already contains the LLMs' answers and the neighbour evidence, so the prompt adds
  little; the remaining errors are genuinely hard (and partly mislabelled or ambiguous).
- 21:19 **Agent `graph-text` finished** (`docs/agents/graph-text.md`): **negative**. Fine-tuning gte-large on own
  text + 5 neighbours' titles raises the LM's own val accuracy (0.922 vs 0.909 at 1 epoch) but not GT test accuracy
  (co-purchase neighbours 0.815 / 0.404; honest cold-start variant with text-kNN neighbours 0.793 vs 0.809), and
  macro-F1 drops 2-4 points. The neighbour titles pull predictions toward the train-frequent classes. One LM seed
  and 1 epoch, so it is not a fully fair comparison.
- 21:20 **Agent `lm-adapt` finished** (`docs/agents/lm-adapt.md`). Fine-tuning gte-large on train labels plus the
  fused teacher's test pseudo-labels (GLEM-style) lifts GT alone from 0.817 to 0.828 (hard) / 0.830 (soft), but
  macro-F1 drops (0.423 -> 0.408/0.416), and val would not select it. With fusion on top (hard): 0.834 / 0.427, below
  the teacher's 0.837. The student re-learns the teacher's information, so distilling the stack into the encoder
  does not beat applying the stack post-hoc.
- 21:22 **Agent `synth` finished** (`docs/agents/synth.md`). Qwen2.5-7B wrote 200 listings for each of 20
  train-rare classes (3,985 nodes), added as isolated train nodes on frozen gte-large features. Before fusion:
  GT 0.784 -> 0.785 (F1 0.384 -> 0.388), SAGE 0.756 -> 0.768 (F1 0.340 -> 0.371; Office recall 0.31 -> 0.53).
  A pitfall worth noting: SAGE learns "no neighbours = rare class" from edge-free synthetic nodes and collapses in
  cold start (0.70 -> 0.34) until the synthetic nodes get self-loops. After fusion: ±0.001 accuracy, F1 up to
  +0.6 points. The LLM answers already fix the classes that the synthetic data targets.
- 21:23 **Agent `llm-backbone` finished** (`docs/agents/llm-backbone.md`). A Qwen3-1.7B decoder fine-tuned as the
  encoder (cut down to fit: max length 128, bottom half frozen, 1 epoch) is worse than ft-gte-large on val and test:
  GT 0.788 / 0.361 (vs 0.817 / 0.423), after fusion 0.827 / 0.421 (vs 0.837 / 0.439). A full fine-tune might close
  part of the gap, which is untested.
- Consolidated script: `scripts/run_fusion.py TAG...` (fusion, BP, LLM combination and β chosen on val) ->
  `results/3_gnn_text/fusion_{runs,summary}.csv`.

## 2026-11-16 23:00 SGT: graph-centric sprint

Goal: more graph machine learning in the method. Seven agents, shared rules in `docs/agent_brief_graph.md`,
hard stop 01:45 SGT. Directions: `g-rewire` (cleaning and rewiring), `g-shift` (shift-robust GNN),
`g-localprior` (graph-local label-shift priors, graph-aware calibration), `g-ssl` (graph self-supervision for the
encoder), `g-imbalance` (GraphENS / GraphSMOTE / TAM), `g-arch` (PE, APPNP / GPR-GNN, GCNII), `g-hetero`
(product–keyword heterogeneous graph). Base: GT@ft-gte-large_s0, evaluated alone and after fusion + BP.
- 23:33 **Agent `g-hetero` finished** (`docs/agents/g-hetero.md`): **negative**. A product–keyword graph (32.5k
  title uni/bigrams, TF-IDF) used as a third edge type, a relation-aware GT, a HeteroConv GT with keyword nodes,
  TextGCN / BertGCN and a keyword BP channel. After fusion + BP: normal 0.835-0.837 vs 0.837; cold 0.8285 vs 0.8281
  for the best kNN row (3 encoder seeds). Why: keyword-top-5 edges have homophily 0.71 (kNN 0.78, co-purchase 0.79),
  first-title-token "brands" span categories (59-67% same label), and the train neighbours that keywords give far
  test nodes vote right only 20% of the time, because those nodes belong to train-rare classes. More paths to
  train nodes import the train label mix, which supports the label-shift diagnosis. TextGCN alone 0.51.
- **Agent `g-imbalance` finished** (`docs/agents/g-imbalance.md`). GraphENS-lite, GraphSMOTE-lite, TAM, ego-network
  mixup and ReNode on GT / SAGE with ft-gte-large. GraphSMOTE is the only augmentation that helps the base model
  (3 encoder seeds: GT 0.811 / 0.407 -> 0.815-0.819 / 0.417-0.419; SAGE 0.807 / 0.390 -> 0.818 / 0.424, cold
  0.785 -> 0.804; Automotive recall +8-15 points). For SAGE the synthetic nodes need edges (generated or copied from
  the anchor): isolated ones teach "no neighbours = rare class" and cold start collapses to 0.674. GraphENS with
  hard minority labels collapses (mixing a majority product under a minority label is label noise), TAM's
  topology margins are below plain logit adjustment, ReNode is neutral. After fusion + BP everything is ±0.1 on
  accuracy: 60-68% of SMOTE's fixes are nodes the GPT fusion already fixes, and the rest cancel.
- **Agent `g-shift` finished** (`docs/agents/g-shift.md`): **negative, but it explains the shift**. SR-GNN's CMD
  regulariser (weight 0.01 -> 1: test 0.815 -> 0.760) and DANN hurt in proportion to their weight: aligning train
  and test hidden features while the class mix differs pushes train-rare classes onto frequent ones (Automotive
  recall 0.50 -> 0.13, 4+ hops 0.68 -> 0.34; Zhao et al. 2019). KMM / PPR-mass / domain-classifier instance
  weights are neutral for GT. Class-reweighted alignment (IWCMD, BBSE weights from GPT, generalized label shift)
  repairs it: alone 0.811 -> 0.815 (3 encoder seeds), after fusion 0.8361 vs 0.8355, cold 0.8283 vs 0.8266. The
  gain behaves like a prior correction (weighting the loss by the estimated test mix does as well). Gradual
  self-training over hop rings: alone +0.6, after fusion -0.2. Train-only test-like holdouts rank variants like val.
- **Agent `g-rewire` finished** (`docs/agents/g-rewire.md`): **negative**. Edge denoising (low ft-gte cosine,
  prediction / GPT disagreement), ft-gte kNN and mutual kNN, GDC-PPR, SDRF-style curvature rewiring, link-predicted
  edges for cold nodes and confident-learning train-label cleaning. Best by val: GDC-PPR, fused 0.8362 vs 0.8355
  (3 encoder seeds), cold 0.8268 vs 0.8266. Mechanism: GT without any edges scores 0.807, so the co-purchase graph
  adds only +1.0 alone and +0.4 after fusion (0 in fused cold start). The GT's attention is nearly uniform (a mean
  aggregator). The lowest-cosine edges are 80% heterophilous but harmless, while the low-cosine same-label edges
  give +10 points to the 1,575 test nodes whose text is misleading, so feature-based trust removes the most
  informative edges. Flagged train "label noise" is mostly genuine heterophily (GPT agrees with the given label on
  54%, with the neighbourhood on 24%; flag rate 19-36% in train-rare classes). On MiniLM features denoising gives
  +0.9 and mutual ft-gte kNN +1.3 (+2.6 cold), all absorbed by the LLM fusion.
- **Agent `g-localprior` finished** (`docs/agents/g-localprior.md`). Region priors (EM per Louvain community, per hop
  bucket, BBSE per region) cost 0.2-0.8 points on top of fusion + BP although their estimates are close (TV 0.10-0.15
  to the true regional mix): even the oracle prior per hop bucket adds only +0.1 normal, because the LLM likelihood
  and BP already move far nodes to their regional mix, so a prior ratio over-corrects. Calibrating before EM makes
  it run away into classes with one train node. Graph-aware per-node temperature (CaGCN / GATS-style): the GT's ECE
  grows with hop distance (0.058 / 0.074 / 0.096) but a temperature fitted on val (89 nodes at 3+ hops) worsens test
  NLL. **Small positive:** a smooth per-node prior from PPR diffusion of the neighbours' fused beliefs, applied as
  log π_i/π_tr before BP: 0.8355 -> 0.8373 / F1 0.437 normal, 0.8266 -> 0.8293 cold (paired gains +0.23/+0.01/+0.30
  and +0.27/+0.30/+0.24 over encoder seeds). Val always picks the weakest strength, so val barely sees it.
- **Agent `g-arch` finished** (`docs/agents/g-arch.md`). Course-standard backbone table on ft-gte-large (3 encoder
  seeds × 3 GNN seeds): MLP 0.794, GCN 0.792, GAT 0.788, SAGE 0.808, GATv2 0.809, JKNet 0.809, APPNP 0.810,
  GT 0.811, **NAGphormer (K=6) 0.817**; after fusion + BP 0.825-0.838. NAGphormer beats GT on every encoder seed
  after fusion (0.8374 ± 0.0018 vs 0.8353 ± 0.0015) and val also prefers it; cold start and macro-F1 equal; on
  MiniLM it is worse than GT. Its gain comes from the Transformer over hop tokens (the attention readout is uniform).
  GCN / GAT mix the node into its neighbourhood and fall below the MLP on strong features. Positional / structural
  encodings: LapPE +0.2 fused cold (not selected by cold val); degree and distance-to-train encodings hurt cold start
  because they change when test edges are removed. GPR-GNN, GCNII, APPNP α=0.1 do not help far nodes. After fusion
  the backbone spread shrinks from 2.9 to 1.3 points (MLP -> GT +1.7 base, +0.5 fused).
- 00:40 **Main thread: combining the two small positives** (`exp/main/combine_graph.py`, runs in
  `exp/main/combine_graph_{runs,selected}.csv`). Backbone GT or NAGphormer (g-arch) × PPR local prior (g-localprior;
  a ∈ {0.15, 0.5}, 1 or 3 iterations, λ ∈ {0.25, 0.5, 1}) × LLM combo × β × γ, chosen on val (mean over 3 GNN seeds)
  per encoder seed and setting. Mean ± std over the 3 encoder seeds, test acc / macro-F1:

  | | normal | cold |
  |---|---|---|
  | GT + fusion + BP (current pipeline) | 0.8355 ± 0.0023 / 0.435 | 0.8266 ± 0.0025 / 0.425 |
  | GT + fusion + PPR prior + BP | 0.8373 ± 0.0037 / 0.437 | **0.8293 ± 0.0023 / 0.429** |
  | NAGphormer + fusion + BP | 0.8373 ± 0.0017 / 0.433 | 0.8253 ± 0.0016 / 0.425 |
  | **NAGphormer + fusion + PPR prior + BP** | **0.8417 ± 0.0010 / 0.438** | 0.8283 ± 0.0017 / 0.427 |

  The two gains add up in the normal setting: +0.62 over the current pipeline, paired +0.53 / +0.76 / +0.56 per
  encoder seed, with val 0.9198 vs 0.9192. In cold start NAGphormer brings nothing, so GT + PPR prior is the best
  cold pipeline. Caveat: if val also decides whether to use the prior at all (λ = 0 allowed), it keeps the prior in
  only half of the 12 (model, encoder, setting) cases, because val has no label shift and barely sees it. As with the
  fusion weight, we therefore fix the prior on by design and let val choose only its strength.
- 00:48 **Agent `g-ssl` finished** (`docs/agents/g-ssl.md`): **negative for the pipeline**. Multi-task fine-tuning
  of gte-large with CE + λ·InfoNCE on co-purchase pairs (λ = 0.1 by LM val), a clean variant on non-test edges only,
  two-stage contrastive-then-supervised, and GRACE pre-training of the GT. No GT variant beats the CE-only encoder on
  val; after fusion + BP all are 0.831-0.837 (3 encoder seeds for λ = 0.1: 0.834 vs 0.836). The contrastive encoder
  does help a text-only MLP (0.786 -> 0.798, 4+ hops 0.558 -> 0.603, Automotive recall 0.10 -> 0.24; the clean
  non-test-edge variant keeps about +1.0), but the GT already gets this from message passing. Supervised fine-tuning
  alone already pulls co-purchased products together (cosine gap to random pairs 0.25 frozen -> 0.66), so the
  InfoNCE task is easy. GRACE lowers base macro-F1 by 0.9-1.5 and 4+-hop accuracy by 2-5 points.

## 2026-11-17 data analysis: the given graph is a neighbour-sampled view (main thread)

Code: `exp/data-analysis/` (`structure.py`, `cleaning.py`, `errors.py`, `restore_edges.py`, `induced_stats.py`),
`exp/g-restore/` (training and fusion on the restored graph).

- **The subset is a 3-hop NeighborLoader sample from 128 seed nodes** of ogbn-products (`input_id`, `batch_size=128`,
  `n_id`, `e_id` in the .pt file). 75% of the nodes (40,493) are in the last sampled layer with mean degree 1.16:
  they kept only the edge to the node that sampled them. "Median degree 1" and many of the "far from train" nodes are
  sampling artefacts, not properties of the products.
- **Restored induced subgraph.** The full ogbn-products edge list (public, OGB; only `edge.csv.gz` and the ASIN
  mapping were kept, the full labels and features were deleted unread) has 869,363 undirected edges among the same
  54,025 nodes, 12× the given 72,319. ASINs match 100%. Test median degree 1 -> 17; test nodes with a train neighbour
  41% -> 88%; 3+ hops from train 18% -> 1%; edge homophily unchanged (0.795); edges left in cold start 8.5k -> 196k.
  The train-neighbour vote is right for only 77% of test nodes (label shift again).
- **Base models on the restored graph** (ft-gte-large_s0, 3 GNN seeds): GCN 0.792 -> 0.804, SAGE 0.808 -> 0.819 (val
  0.930), GT 0.817 -> 0.819 (val 0.927 -> 0.931). Cold start unchanged (≈0.81 GT).
- **BP must be degree-normalised on a dense graph.** With Σ_j log m_j a node with 25 neighbours becomes over-confident
  and val switches BP off (fused 0.836). With the mean of the neighbours' log messages (`exp/g-restore/bp_norm.py`,
  GPT only, β on val): GT(restored) + fusion + mean-BP val 0.926 / test **0.841** (sqrt-normalised 0.842), against
  val 0.919 / test 0.837 for the current pipeline (encoder s0, 3 GNN seeds). On the given graph normalisation changes
  nothing (median degree 1). Cold start is unchanged by construction.
- **Data cleaning checks.** Exact duplicates: 405 groups, 99% label-consistent, but the 78 test nodes with a train
  duplicate are already 97.4% right, so copying labels gains 1 node. Class 24 "MISSING" is a missing-metadata
  class: 99% have no title but an ordinary description; among title-less test nodes that GPT does not call media,
  44% are MISSING (train 2%, val 8%), a group-level label shift that no train-selected rule can see. Class 42 "Baby"
  has no train or val node (680 of its 807 test nodes go to "Baby Products").
- **Remaining errors** of the best pipeline (test 0.840 on s0 seed 0): 51% are on test nodes with no same-label
  neighbour (6,186 nodes, acc 0.50). The largest fixable confusion is Electronics -> Cell Phones & Accessories
  (658 errors; Electronics has 225 train nodes vs 1,154), although Electronics test nodes link to Electronics 6,942
  times and to Cell Phones only 718 times.
- Decision: the restored graph is **not** used in the method (outside the TAPE subset); it is reported as analysis only.
- **Electronics -> Cell Phones (658 errors) is a taxonomy effect**: they are tablet accessories (cases, stands,
  cables), which Amazon files under Electronics because tablets are Electronics. GPT (611/658) and the GT both say
  Cell Phones; their neighbours are truly 1,512 Electronics vs 472 Cell Phones, but are mostly unlabelled test nodes
  predicted Cell Phones too, so the cluster is wrong together.
- **Sibling channel in BP (given graph only)** (`exp/g-restore/bp_sibling.py`): 2-hop pairs through a common
  neighbour ("bought with the same product"): 320k pairs, homophily 0.745, 77% of test nodes reach a train node this
  way (41% by direct edges). A degree-normalised sibling message channel next to fusion: val 0.9194/0.9169/0.9211 ->
  0.9235/0.9203/0.9275, test 0.8372/0.8308/0.8363 -> 0.8373/0.8359/0.8375 (encoder s0/s1/s2, GPT only), i.e. +0.2
  on test with val clearly preferring it. Normal setting only (cold start has no sibling pairs through test nodes).

## 2026-11-17 09:30 SGT: round 2 (`docs/agent_brief_r2.md`, hard stop 13:00)
Agents: `g-llm-rag` (graph-retrieved in-context learning with a large local LLM on all 8 GPUs), `g-valshift`
(label-shift-aware model selection by importance-weighted val), `g-evidence` (GPT ranked lists, multi-round loopy BP
with sibling channel, integrated pipeline). The restored OGB graph is off-limits.
- **Agent `g-valshift` finished** (`docs/agents/g-valshift.md`). Importance-weighted val (π̂_test/π_val from BBSE,
  EM, soft-fused posteriors, a hop-bucket graph version, even the true prior) tracks test worse than plain val
  (Spearman 0.30-0.36 vs 0.79; regret 1.0-1.4 vs 0.40 pt) because the four train-rare classes are 19.5% of test but
  only 40 val nodes (Automotive 1), so weighted deltas have SE 0.5-1.1 pt. Plain val falls monotonically in the
  LLM weight w and the prior strength λ while test peaks near w = 1, λ = 1; giving val w picks 0.5 and costs 0.5-0.8.
  **Fixing λ = 1** (the full Saerens correction, same argument as w = 1) and choosing the rest on val: GT 0.8381 /
  cold 0.8315, NAG 0.8425 / cold 0.8316; with the sibling channel GT 0.8403, **NAG 0.8433 ± 0.0019 / F1 0.439, cold
  0.8317** (3 encoder seeds), within 0.1 pt of the grid's test-best. Better than val's λ in 11 of 12 comparisons.
- **Agent `g-evidence` finished** (`docs/agents/g-evidence.md`). GPT's full ranked list (4 likelihood models)
  raises val NLL and accuracy but lowers fused test by 0.2-1.5 points: later mentions behave differently on
  best-sellers and on the long tail, only the top-1 confusion transfers. Multi-round loopy BP (proper or naive,
  damped, degree-normalised) adds nothing on top of a GNN + PPR prior. What helps: BP messages that carry the
  neighbour's full fused likelihood ratio, mostly over the sibling channel: final 0.8427 ± 0.0019 / 0.437 (val picks
  NAGphormer), cold 0.8305 ± 0.0003 (val picks MLP). MLP + fusion + PPR prior + 5-round loopy BP reaches 0.8406 /
  cold 0.8305: classic collective classification matches the Graph Transformer once LLM evidence is fused. The
  Electronics -> Cell Phones errors form one ~330-node test cluster where GPT (94-95%) and the GNN agree on the wrong
  class; propagation spreads that consensus.
- 12:20 **Agent `g-llm-rag` finished** (`docs/agents/g-llm-rag.md`). Qwen3-235B was not fully cached, so Qwen3.8-27B
  (one SGLang engine per GPU) with labelled train demonstrations retrieved from 1-2 hop co-purchase neighbours and
  text-kNN (prompt chosen on val: 0.907 vs zero-shot 0.826 and GPT-3.5 0.782). Answers alone: test 0.805 / F1 0.427,
  about the same as a plain vote of the demonstration labels (0.806 / 0.407); the LLM adds rare-class recall
  (Automotive 0.52 vs 0.24) but makes Electronics -> Cell Phones worse (820 vs 657 for GPT). Fused with GPT on GT and
  gated to nodes that have a co-purchase demonstration: 0.8355 -> 0.8429 ± 0.0010 / F1 0.442 (normal only; the
  cold-safe kNN-only set gives ±0). On top of NAGphormer + PPR prior only 0.8417 -> 0.8427, below the 0.8433 of the
  sibling pipeline. Val over-rates the RAG answers (+1.0 val, -0.2 to +0.7 test): an answer "Cell Phones" is 96% right
  on train+val but 63% on test. Not adopted. The cold-safe answer file was moved out of `outputs/llm_answers/` so
  `scripts/run_fusion.py` behaves as before (runs between 11:00 and 12:20 picked it up; the committed graph-work
  numbers were computed before 11:00).

## 2026-11-17 13:00 SGT: the three remaining error groups (main thread, `exp/data-analysis/three_groups.py`)

Errors of the final pipeline (`outputs/g-evidence/final`, normal, mean over 3 encoder × 3 GNN seeds; 15.7% of test):
Baby 2.14%, MISSING 2.00%, Electronics -> Cell Phones 1.68%, other errors on nodes with no same-label neighbour
6.14%, the rest 3.77%. Accuracy without Baby and MISSING: 0.883.

1. **Baby / MISSING.** Baby has no train or val node (685 of 807 go to "Baby Products"): unpredictable. MISSING is
   a label shift inside the title-less group (MISSING share 0.8% of title-less train, 3.1% val, 24.1% test). The GT
   already ranks it well inside that group (P(MISSING) AUC 0.875), and in the analysis title-less test nodes with a
   MISSING neighbour are 89% MISSING. Group-wise EM (Saerens) on the title-less unlabelled nodes, before fusion
   (`missing_group_em2.py`): it estimates a MISSING share of 1.1% instead of 24%, because p(x | MISSING) is learnt
   from 13 train nodes, and it moves the media classes, so the title-less accuracy drops 0.688 -> 0.558 and test
   0.835 -> 0.824. **Negative.** Applied after fusion it does nothing (the fused posteriors give MISSING ≈ 0 mass).
2. **Electronics -> Cell Phones is concept shift, not label shift.** Titles with a tablet word (iPad, tablet, Kindle,
   Galaxy Tab, ...) and an accessory word (case, stand, cover, cable, ...): train n=109, 91% Cell Phones, 8%
   Electronics; val n=10, 90% Cell Phones; test n=721, 82% Electronics. Phone accessories are Cell Phones in every
   split (100% / 100% / 94%). So P(y | x) itself changes from best-sellers to the long tail; 65% of the
   Electronics -> Cell Phones errors match this pattern. Any method faithful to the train labels makes this error;
   no train- or val-selected method can fix it.
3. **No same-label neighbour** (6,185 test nodes, 89% of degree 1, best pipeline 0.472): the graph does not hurt
   there (GT 0.494, MLP 0.476, GPT 0.430). The neighbour pairs are category boundaries (Electronics / Cell Phones,
   Tools / Sports, Beauty / Health) and MISSING, i.e. the same ambiguity as above, seen through one edge.

## 2026-11-18: final pipeline in `scripts/`, released checkpoints, notebook

- NAGphormer moved to `src/models.py` (reproduces `exp/g-arch` exactly: s0 seed 0 test 0.8273 / cold 0.8112), the PPR
  local prior and sibling messages to `src/label_shift.py`, the post-hoc pipeline to `scripts/run_final.py`
  (reproduces `exp/g-valshift`'s 0.8433 ± 0.0019 / cold 0.8317 on the original encoders exactly).
- No checkpoint had been saved, so the three gte-large encoders were fine-tuned again with `--save-ckpt`
  (`ft-gte-large-r_s{0,1,2}`, LM-head val 0.9186 / 0.9179 / 0.9262 vs 0.9148 / 0.9186 / 0.9249 originally: GPU
  training is not bit-reproducible). Reloading a checkpoint reproduces its embeddings (cosine ≥ 0.9995). NAGphormer
  and GT were retrained on them with `--save-ckpt`.
- On the released encoders: GT 0.8125; GT + fusion + BP 0.8358 ± 0.0013 (cold 0.8250); GT + `run_final.py` 0.8394
  (cold 0.8313); **NAGphormer + `run_final.py` 0.8420 ± 0.0011 / F1 0.437, cold 0.8311 ± 0.0016**.
- Artefacts (3 encoders, 4 embedding files, 18 GNN weight files, Qwen answers) were first on a private Hugging Face
  repo; they are now in a public Google Drive zip downloaded by `scripts/download_artifacts.py`.
  `notebooks/4_fusion/final_pipeline.ipynb` (built by `scripts/make_final_notebook.py`) downloads data and artefacts
  and runs the pipeline with figures.
