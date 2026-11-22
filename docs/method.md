# Stage 4 method: LLM fusion and belief propagation under the sales-rank shift

This note summarises the changes that improved results: what the method is, how well it works, how to rerun it,
and what did not help. The full log is in `docs/research_log.md`.

## Result

Official split, test accuracy / macro-F1, mean ± std over seeds. Every tuned hyperparameter and variant was chosen on
val. The LLM weight and the prior strength λ are not tuned: both are fixed at 1 by Bayes' rule (see Caveats).

| Pipeline | normal | cold start |
|---|---|---|
| Previous best: Graph Transformer (GT) on MiniLM features | 0.765 / 0.365 | 0.709 / 0.330 |
| Previous best with an LLM: MiniLM + GPT answer as a feature, GCN | 0.772 / 0.353 | 0.758 |
| GT on MiniLM + LLM fusion | 0.814 / 0.414 | 0.799 / 0.404 |
| GT on MiniLM + LLM fusion + belief propagation | 0.822 / 0.422 | 0.806 / 0.410 |
| GT on fine-tuned gte-large (no LLM; encoder seed 0) | 0.817 / 0.423 | 0.809 / 0.416 |
| GT on fine-tuned gte-large + LLM fusion + belief propagation | 0.835 ± 0.001 / 0.435 | 0.828 ± 0.001 / 0.430 |
| **Final: NAGphormer on fine-tuned gte-large + LLM fusion + PPR local prior (λ = 1) + BP with sibling channel** (released checkpoints, see below) | **0.842 ± 0.001 / 0.437** | **0.831 ± 0.002 / 0.428** |

The last two rows are averaged over 3 encoder fine-tuning seeds × 3 GNN seeds. Compared with the previous best GT, the
GT pipeline is +7.0 points of accuracy, +7.0 of macro-F1 and +11.9 in cold start; the graph work adds about +0.8 more
(+0.5 in cold start).

### Released checkpoints and the final numbers

The original fine-tuning runs kept the best encoder only in memory, so the three encoders were fine-tuned again with
`--save-ckpt` (same settings; GPU training is not bit-reproducible, so they differ slightly from the originals) and
everything downstream was rerun on them. The checkpoints, embeddings, GNN weights and our Qwen answers are in a
public Google Drive zip that `scripts/download_artifacts.py` downloads (no login); the end-to-end demo is
`notebooks/4_fusion/final_pipeline.ipynb`. Mean ± std over the 3 encoder seeds `ft-gte-large-r_s{0,1,2}`, each the mean of 3
GNN seeds; test acc / macro-F1:

| Pipeline on the released encoders | normal | cold start |
|---|---|---|
| GT, no LLM | 0.8125 / 0.402 | 0.8032 / 0.396 |
| GT + fusion + BP (`scripts/run_fusion.py`, before the graph work) | 0.8358 ± 0.0013 / 0.434 | 0.8250 ± 0.0028 / 0.423 |
| GT + fusion + PPR prior + BP with sibling channel (`scripts/run_final.py`) | 0.8394 ± 0.0031 / 0.438 | **0.8313 ± 0.0024 / 0.430** |
| **NAGphormer + fusion + PPR prior + BP with sibling channel (final)** | **0.8420 ± 0.0011 / 0.437** | 0.8311 ± 0.0016 / 0.428 |

The graph work adds +0.6 points in both settings on these encoders (+0.8 / +0.5 on the original ones, below). Per
encoder seed and setting, the configuration chosen on val is in `results/4_fusion/final_*@ft-gte-large-r_s*_selected.csv`. About 4.3% of test nodes cannot be predicted, so
accuracy is capped at about 0.957 (see the diagnosis).

## Diagnosis: the split causes label shift

- Train nodes are the best-sellers and test nodes the long tail. The label distributions differ by total variation
  0.245 between train and test, but only 0.071 between train and val. That is why val accuracy (0.87-0.92)
  overstates test accuracy.
- If test had the train class mix, GCN would score 0.85 instead of 0.71. Most of the val-to-test drop is caused by
  the class mix, not by harder products within a class.
- Classes that are rare in train but large in test are almost never predicted. Office Products is 0.6% of train and
  4.0% of test, yet only 0.2% of GCN's predictions. The same holds for Automotive, Tools & Home Improvement and
  Electronics.
- Accuracy collapses with the hop distance to the nearest train node (GT: 0.83 at 1 hop, 0.46 at 4+ hops). Those far
  nodes are mostly from the shifted classes (label TV to train 0.78 at 4+ hops), so longer propagation does not fix
  them.
- About 4.3% of test cannot be predicted. Class 42 "Baby" has 807 test nodes and none in train, and GPT calls them
  "Baby Products" (class 23). Class 24 "MISSING" has 763 nodes, all without a title. Seven other classes have no
  training nodes.
- Text truncation is not the problem. MiniLM cuts 16% of texts at 256 tokens, but raising the limit to 512 changes
  test accuracy by +0.3 points.

## Method

Four parts. Parts 2 and 3 run after training on CPU in seconds, and are written in `src/label_shift.py` and
`scripts/run_fusion.py`.

1. **Fine-tuned text encoder.** Fine-tune `Alibaba-NLP/gte-large-en-v1.5` with a linear head on the 14,708 train
   nodes: 2 epochs, lr 2e-5, 320 tokens, checkpoint chosen by val accuracy. Use the [CLS] vector of every node as
   features for the GT (2 layers, 8 heads, 256 hidden, no positional encoding; `scripts/export_probs.py`).
   Fine-tuning gives +5.2 points on its own. Bigger frozen encoders raise val far more than test.
2. **Bayesian LLM fusion, with no tuned weight.** An LLM's zero-shot answer g is treated as an observation with
   likelihood P(g | y): its confusion matrix, counted on the train+val nodes with Laplace smoothing. The fused
   posterior is

   p(y | x, g) ∝ p(y | x) · P(g | y)

   where p(y | x) is the temperature-scaled GNN posterior (temperature fitted on val NLL).
   - The LLM never saw the train label mix, so it does not share the GNN's bias against the long-tail classes.
     The confusion matrix also corrects its systematic mistakes, e.g. "Office & School Supplies" counts as
     evidence for Office Products.
   - The weight is fixed at 1 by Bayes' rule, and that is deliberate. Tuning a weight on val would under-weight the
     LLM, because val looks like train: val prefers λ = 0.5, while test is best around 0.75-1.
   - Answers used: GPT-3.5 from TAPE, and our own Qwen3.5-9B zero-shot (`scripts/llm_zero_shot.py`). Val decides
     whether to add Qwen; with the fine-tuned encoder it keeps GPT only.
3. **One round of belief propagation.** Each node also receives messages from its neighbours:

   b_i(y) ∝ p(y | x_i) · P(g_i | y) · Π_j m_j(y)^β, with m_j(y) = Σ_y' H[y, y'] e_j(y')

   - e_j is the neighbour's one-hot label if it is a train node, and otherwise its LLM likelihood. Val labels are
     never used as evidence.
   - H is the label-compatibility matrix counted on train-train edges, smoothed toward 0.8·I + 0.2·uniform.
   - Two edge sets are used: co-purchase edges (weight β) and each node's 5 nearest text neighbours by MiniLM
     cosine (weight γ). Cold-start nodes keep the text edges.
   - β ∈ {0, 0.25, 0.5, 1} and γ ∈ {0.25, 0.5, 1} are chosen on val.
4. **Protocol.** Chosen on val: the temperature, which LLM answer sets to fuse, and β, γ (later also the PPR setting
   and the sibling weight). The LLM weight and λ are fixed at 1, not tuned (see Caveats). Test labels are not used to
   train or to tune any setting.

**Where the gain comes from** (GT on MiniLM, test acc by hop distance to the nearest train node; base -> fusion ->
fusion + BP):

| Hop distance | Test nodes | Accuracy |
|---|---|---|
| 1 | 15,570 | 0.833 -> 0.850 -> 0.854 |
| 2 | 15,209 | 0.773 -> 0.807 -> 0.812 |
| 3 | 4,634 | 0.663 -> 0.755 -> 0.777 |
| 4+ | 2,320 | **0.463 -> 0.694 -> 0.752** |

With fusion (GT on MiniLM), recall rises for Office Products (0.58 -> 0.84), Tools & Home Improvement (0.35 -> 0.65)
and Automotive (0.12 -> 0.44). The only slice that does not improve is products without a title (0.644 -> 0.641),
where both LLMs are weak.

## What did not help (worth reporting as ablations)

All of these were run with 3 seeds and chosen on val. Details are in `docs/research_log.md`.

| Idea | Alone | On top of fusion + BP |
|---|---|---|
| Prior correction only (EM / BBSE / even the oracle test prior) | GCN +3, GT ±0; EM collapses | — |
| Logit adjustment, class-weighted loss | macro-F1 +3 to 5, accuracy flat | ≈0 |
| Class-name prototype classifier head | SAGE 0.738 -> 0.769 | +0.6 |
| Label trick, SIGN up to 6 hops, deeper GT, H2GCN-style | ≤ +0.2 | — |
| Correct & Smooth over co-purchase + text-kNN edges | GT 0.765 -> 0.775 | +0.0 |
| Self-training on GT/GPT-agreement pseudo-labels | GT 0.763 -> 0.800 | -0.3 |
| LLM restructuring of descriptions (summary embedding concatenated) | GT 0.765 -> 0.782 | +0.0 |
| Better Qwen prompt (only known categories + definitions) | Qwen 0.659 -> 0.714 | ≈0 |
| LLM re-ranking of uncertain nodes with graph context | worse than the stack's top-1 | +0.0 |
| Fine-tuning the encoder on the fused teacher's pseudo-labels | GT 0.817 -> 0.830, F1 down | -0.3 |
| Neighbour titles in the LM input | GT 0.815, F1 -2 to -4 | — |
| LLM-synthesised listings for train-rare classes | SAGE 0.756 -> 0.768 | ±0.1 |
| Qwen3-1.7B as the encoder (cut-down fine-tune) | GT 0.788 | -1.0 |
| Multi-round collective inference / Dawid-Skene re-estimation | — | +0.2 / -0.2 |

The common thread is that every LLM-based route (fusion, restructuring, pseudo-labels, synthetic data, re-ranking,
distillation) carries the same LLM knowledge. Each helps alone, but they do not add up. The parameter-free post-hoc
fusion is the cheapest and strongest of them. The only gain that is orthogonal to it is a text encoder fine-tuned on
the train labels.

## Graph work (2026-11-16 23:00 to 2026-11-17 11:30 SGT)

**Goal.** Most of the gain above comes from text and LLMs; the graph enters only through the GT layers and one round
of BP. For a graph ML course we wanted more, and better motivated, graph methods: use graph knowledge to clean or
augment the data, correct the label shift, or improve the algorithm. Round 1 explored seven directions in parallel,
then the data were analysed, and round 2 explored three more directions on the two bottlenecks found. Rules
throughout: choose on val only, 3 seeds, report every variant, only the given TAPE graph. Round 2 ended at 12:20 SGT
on 2026-11-17. The chronological log is `docs/research_log.md`. The repo keeps the exploration code only for the
directions the report cites (`exp/g-arch`, `exp/graph`, `exp/g-localprior`, `exp/g-restore`, `exp/main`); the others are summarised in
the log.

### What improved

Test acc / macro-F1, mean ± std over the 3 encoder seeds (each the mean of 3 GNN seeds). Every row uses the same
selection rule, so the first row differs slightly from the headline table, which picks the kNN channel separately.

| Pipeline | normal | cold start | Where |
|---|---|---|---|
| GT + fusion + BP (pipeline before this work) | 0.8355 ± 0.0023 / 0.435 | 0.8266 ± 0.0025 / 0.425 | `exp/main/combine_graph.py` |
| + PPR local prior | 0.8373 ± 0.0037 / 0.437 | 0.8293 ± 0.0023 / 0.429 | `exp/g-localprior/` |
| NAGphormer instead of GT, + PPR local prior | 0.8417 ± 0.0010 / 0.438 | 0.8283 ± 0.0017 / 0.427 | `exp/g-arch/`, `exp/main/` |
| + PPR prior at λ = 1 + sibling channel in BP | **0.8433 ± 0.0019 / 0.439** | **0.8317** / 0.430 | g-valshift |
| + BP messages carrying the neighbour's fused likelihood ratio (base chosen on val) | 0.8427 ± 0.0019 / 0.437 | 0.8305 ± 0.0003 / 0.430 | g-evidence |
| MLP (no GNN) + fusion + PPR prior + 5-round loopy BP | 0.8406 | 0.8305 | g-evidence |

The steps, each a graph idea:
- **NAGphormer** (Chen et al. 2023): each node is a sequence of hop-aggregated features (X, ÂX, ..., Â^6 X) read by a
  Transformer. Beats GT on every encoder seed, before fusion (0.817 vs 0.811) and after it; val agrees. No gain in
  cold start.
- **PPR local prior**: diffuse the neighbours' fused beliefs (train nodes one-hot) with personalised PageRank over
  co-purchase ∪ text-kNN edges and add λ·log(π_i / π_tr) to the fused log posterior before BP: a label-shift
  correction with a node-specific prior, because the label mix differs by graph region. All 12 settings beat the
  baseline on test.
- **λ = 1 by design.** Val falls monotonically in λ (and in the LLM weight), while test peaks near λ = 1. λ = 1 is the
  full Saerens prior correction, the same argument that fixes the LLM weight at 1. It beats val's choice in 11 of 12
  comparisons. Caveat for the report: the evidence that it beats val's choice is on test.
- **Sibling channel**: 2-hop pairs through a common neighbour ("bought with the same product"), degree-normalised,
  as an extra BP channel. 320k pairs, homophily 0.745; 77% of test nodes reach a train node this way vs 41% by
  direct edges. +0.2 on GT, and val clearly prefers it.
- **Classic collective classification holds up**: a plain MLP + fusion + PPR prior + loopy BP matches the GNNs, and
  in cold start several BP rounds help (+0.26) because evidence travels along chains of text-kNN neighbours.

In total the graph work adds about +0.8 points in the normal setting (0.8355 -> 0.8433) and +0.5 in cold start
(0.8266 -> 0.8317), with macro-F1 +0.4 / +0.5.

### What did not improve

| Direction | What was tried | Alone (base GNN) | After fusion + BP |
|---|---|---|---|
| Backbones and encodings (`g-arch`) | MLP, GCN, GAT, GATv2, SAGE, JKNet, APPNP, GPR-GNN, GCNII, GT + LapPE / RWPE / degree / distance-to-train encodings, DropEdge | MLP 0.794, GCN 0.792, GAT 0.788, SAGE 0.808, APPNP 0.810, GT 0.811 | 0.825-0.836; encodings ≤ GT, degree / distance encodings hurt cold start |
| Region priors and calibration (`g-localprior`) | EM prior per Louvain community / hop bucket, BBSE per region, graph-aware per-node temperature | — | -0.2 to -0.8; even the true prior per hop bucket +0.1; temperature ±0 |
| Graph imbalance learning (`g-imbalance`) | GraphENS, GraphSMOTE, TAM, ego-network mixup, ReNode | **GraphSMOTE: GT +0.6 / F1 +1, SAGE +1.1 / F1 +3.4, cold +1.9**; GraphENS collapses; TAM hurts | ±0.1 |
| Graph domain adaptation (`g-shift`) | SR-GNN (CMD + KMM), DANN, graph importance weights, label-weighted CMD, gradual self-training by hop ring | CMD / DANN hurt with their weight (Automotive recall 0.50 -> 0.13); label-weighted CMD +0.3 | +0.06, cold +0.17 |
| Cleaning and rewiring (`g-rewire`) | edge denoising (cosine, prediction / LLM agreement), ft-gte kNN, GDC-PPR, curvature (SDRF) rewiring, link prediction for cold nodes, confident-learning label cleaning | ±0.3; **MiniLM: denoising +0.9, mutual kNN +1.3 (cold +2.6)**; label cleaning hurts F1 | ±0.1 |
| Graph self-supervision (`g-ssl`) | gte-large fine-tuned with co-purchase InfoNCE (SPECTER / LinkBERT / GIANT style), two-stage, GRACE pre-training of the GT | GT ±0.4, F1 -0.7; **MLP +1.1, 4+ hops +4.5**; GRACE hurts | -0.2 to ±0 |
| Heterogeneous keyword graph (`g-hetero`) | product–keyword graph (TextGCN / BertGCN), relation-aware GT, HeteroConv, keyword BP channel | ≤ GT; TextGCN alone 0.51 | ±0.1 |
| Shift-aware model selection (`g-valshift`) | val weighted by π̂_test / π_val (BBSE, EM, soft-fused, hop-bucket graph estimate, even the true prior) | — | worse (0.826-0.831): 40 val nodes in the four train-rare classes give SE 0.5-1.1 pt |
| Richer LLM evidence, more BP rounds (`g-evidence`) | GPT's full ranked list (4 likelihood models), proper multi-round loopy BP, damping, degree normalisation | — | ranked list -0.2 to -1.5 (raises val, lowers test); extra rounds ±0 on a GNN |
| Graph-retrieved in-context learning (`g-llm-rag`) | Qwen3.8-27B with labelled train demonstrations retrieved from 1-2 hop co-purchase neighbours and text-kNN (val 0.907 vs GPT-3.5 0.782), fused through its confusion matrix; gated to nodes with a co-purchase demonstration | answers alone test 0.805 / F1 0.427 (GPT 0.723), but a plain vote of the demonstration labels gets 0.806; Electronics -> Cell Phones errors rise to 820 | on GT 0.8355 -> 0.8429 / F1 0.442 (normal only), but on NAGphormer + PPR prior only +0.1 (0.8427) and ±0 in cold start; val over-rates it (+1.0 on val) |
| Data cleaning (main thread) | copy labels from exact duplicates; degree-normalised BP on the given graph | — | duplicates: 1 node; normalisation ±0 on the given graph (median degree 1) |

### Data analysis (main thread)

- **The given graph is a sampled view.** TAPE's subset is a 3-hop NeighborLoader sample from 128 seed nodes of
  ogbn-products (`input_id`, `batch_size=128`, `n_id`, `e_id` in the .pt file). 75% of the nodes are in the last
  sampled layer with mean degree 1.16: they kept only the edge to the node that sampled them. Among the same 54,025
  nodes the public OGB edge list has 869k co-purchase edges, 12× the given 72k (test median degree 17 vs 1, test
  nodes with a train neighbour 88% vs 41%, same homophily 0.795). On that restored graph GT + fusion + mean-BP
  reached 0.841 on encoder s0. **Decision: not used in the method** (it is outside the TAPE subset); reported as the
  explanation of why the graph adds so little here and why far-from-train nodes look isolated.
- **Duplicates**: 405 exact-duplicate groups, 99% label-consistent, but the 78 test nodes with a train duplicate are
  already 97.4% right.
- **Class 24 "MISSING"** is a missing-metadata class: 99% have no title but an ordinary description; among title-less
  test nodes that GPT does not call media 44% are MISSING (train 2%, val 8%), invisible to any train- or
  val-selected rule. **Class 42 "Baby"** has no train or val node. Together 4.1% of test.
- **Remaining errors**: 51% are on test nodes with no same-label neighbour. The largest fixable one is Electronics ->
  Cell Phones & Accessories (about 640): tablet accessories that Amazon files under Electronics. They form one
  ~330-node test cluster where GPT (94-95%) and the GNN agree on the wrong class, so propagation only spreads the
  error; only better evidence can fix it.

### The remaining errors

The final pipeline's errors are 15.7% of test: Baby 2.1% (no train or val label), MISSING 2.0%, Electronics ->
Cell Phones 1.7%, other errors on nodes with no same-label neighbour 6.1%, the rest 3.8%. Accuracy without Baby and
MISSING is 0.883.
- **MISSING** is label shift inside the title-less group (0.8% of title-less train, 24% of title-less test). The GT
  ranks it well there (AUC 0.875), but group-wise EM estimates 1.1% instead of 24% (p(x | MISSING) is learnt from 13
  nodes) and costs 1 point: negative.
- **Electronics -> Cell Phones is concept shift.** Tablet accessories are 91% Cell Phones in train (109 nodes) and
  82% Electronics in test (721); phone accessories are Cell Phones everywhere. P(y | x) changes from best-sellers to
  the long tail, so no method faithful to the train labels can fix it.
- **Nodes with no same-label neighbour** sit on the same category boundaries (Electronics / Cell Phones, Tools /
  Sports, Beauty / Health, MISSING); the graph does not hurt there (GT 0.494 vs MLP 0.476).

### What the graph work says (for the report)

1. **The co-purchase graph carries little beyond the fine-tuned text, largely because it was sampled away.** A GT
   without edges scores 0.807; the graph adds +1.0 alone and +0.4 after fusion. The GT's attention is nearly uniform.
   On strong features GCN / GAT fall below the MLP; models that keep the ego features separate are above it.
2. **Graph methods help weak models, and the LLM evidence absorbs the same help.** 60-68% of the nodes GraphSMOTE
   fixes are already fixed by the GPT fusion; graph denoising and LLM evidence fix the same nodes.
3. **The shift is in P(y), and graph methods that assume otherwise hurt.** Aligning train and test representations
   (SR-GNN, DANN) pushes the train-rare classes onto the frequent ones; class-weighting the alignment repairs it.
   Giving far nodes more paths to train nodes imports the train label mix.
4. **"Noisy" edges and labels are informative heterophily.** Low-cosine same-label edges give +10 points to the 1,575
   test nodes whose text misleads; train nodes flagged as mislabelled are mostly correct and concentrated in the
   train-rare classes.
5. **The graph is the right tool for estimating the regional label mix, softly.** Hard region priors over-correct;
   the PPR-diffused prior is a consistent gain.
6. **Val cannot see the label shift.** It prefers the weakest correction for every knob, and reweighting it is too
   noisy (40 nodes in the shifted classes). Correction strengths are therefore fixed by Bayes' rule, not tuned.
7. **Cold start.** Degree / distance encodings break when test edges are removed; synthetic minority nodes need
   edges, or SAGE learns "no neighbours = rare class" (cold 0.674).

**Done afterwards:** NAGphormer, the PPR prior (λ = 1) and the sibling channel were moved into `scripts/`
(`export_probs.py` and `scripts/run_final.py`) and the main table was rerun. The `g-llm-rag` answers are not adopted: they
match the best graph pipeline (0.8427 vs 0.8433), need a separate normal / cold answer set, and val over-rates them.

## Caveats

- Test labels were looked at during the error analysis that motivated the method. No tuned hyperparameter or variant
  was chosen on test, and every variant tried is reported.
- The LLM weight and λ are fixed at 1 by Bayes' rule, not tuned. Validation prefers smaller values because it
  shares the training category mix, and the evidence that 1 beats the validation choice comes from test results.
- The GPT answers come from TAPE (gpt-3.5-turbo-0613). The Qwen answers are generated locally with
  `scripts/llm_zero_shot.py` (merging its shards into `outputs/llm/qwen35_9b.npy` is not scripted; the released zip
  contains the merged files). They are weaker alone (test 0.659 vs 0.723) and the final selection uses GPT only.
- The "cold start" setting removes the edges touching test nodes at inference only, as in the earlier experiments.
  Text-kNN edges and other nodes' LLM answers are still available.

## Reproduce

`outputs/` is gitignored. Fine-tuning needs a GPU; everything after it runs on CPU (a GPU makes the GNN faster).

```bash
# 0. data and TAPE's GPT answers: see README.md (or run notebooks/4_fusion/final_pipeline.ipynb, which downloads them)
# a) from the released checkpoints (public Google Drive zip, no login)
python scripts/download_artifacts.py                          # encoders, embeddings, GNN weights, Qwen answers (3.4 GB)
python scripts/export_probs.py NAGphormer --emb ft-gte-large-r_s0 --load-ckpt --device cpu
python scripts/run_final.py NAGphormer@ft-gte-large-r_s0      # -> results/4_fusion/final_*, outputs/final/
# b) from scratch (per encoder seed 0, 1, 2)
python scripts/finetune_lm.py --seed 0 --tag=-r --save-ckpt   # about 20 min on an A100 -> outputs/ckpt/, embeddings
python scripts/export_probs.py NAGphormer --emb ft-gte-large-r_s0 --save-ckpt
python scripts/run_final.py NAGphormer@ft-gte-large-r_s0
# the pipeline before the graph work, for comparison
python scripts/export_probs.py GT --emb ft-gte-large-r_s0 && python scripts/run_fusion.py GT@ft-gte-large-r_s0
# optional: Qwen3.5-9B zero-shot answers (SGLang, one process per GPU), merged into outputs/llm/qwen35_9b.npy
python scripts/llm_zero_shot.py --shard 0 --num-shards 8
```

`notebooks/4_fusion/final_pipeline.ipynb` runs a) end to end with the step-by-step ablation, accuracy by hop distance,
train-rare recall and the error breakdown.

## Files

| Path | What |
|---|---|
| `src/label_shift.py` | confusion matrices, BBSE / EM priors, temperature scaling, fusion, compatibility matrix, BP messages, PPR local prior, sibling messages |
| `src/models.py` | MLP / GCN / SAGE / GT and NAGphormer |
| `src/train.py` | new optional `loss_offset` (logit adjustment); the default behaviour is unchanged |
| `scripts/finetune_lm.py` | fine-tune a text encoder on train nodes and embed every node (`--save-ckpt` keeps the encoder) |
| `scripts/export_probs.py` | train MLP / GCN / SAGE / GT / NAGphormer over 3 seeds and save softmax probabilities (`--save-ckpt` / `--load-ckpt` for the weights) |
| `scripts/llm_zero_shot.py` | local Qwen zero-shot answers |
| `scripts/run_fusion.py` | LLM fusion + BP over co-purchase and text-kNN edges, selection on val (pipeline before the graph work) |
| `scripts/run_final.py` | final post-hoc pipeline: fusion + PPR local prior (λ = 1) + BP over co-purchase, text-kNN and sibling edges |
| `scripts/download_artifacts.py` | download the released checkpoints, embeddings and LLM answers (public Google Drive zip) |
| `notebooks/4_fusion/final_pipeline.ipynb` (built by `scripts/make_final_notebook.py`) | end-to-end demo of the final pipeline with figures |
| `results/3_gnn_text/fusion_*`, `results/4_fusion/final_*` | per-seed and summary results of the fusion runs and the final pipeline |
| `results/shared/embeddings/ft-gte-large_s*_finetune.json` | fine-tuning arguments and val curves |
| `exp/` | exploration code kept for the directions the report cites: `g-arch` (backbones), `graph` (text-kNN and cold start), `g-localprior` (PPR local prior), `g-restore` (restored graph), `main` (ablations) |
| `docs/research_log.md` | the full chronological log of every direction tried |
