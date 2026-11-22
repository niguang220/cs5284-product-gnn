# LLM-Assisted Product Categorisation on a Co-purchase Graph

Group 4 project for CS5284 Graph Machine Learning, NUS, Semester 1 AY2026/27.

Team: Lee En Yi Javier, Cheok Le Yi, Faizatuzzahrah Rahmaniah, He Yufan, Zhao Xuanlei.

## Overview

Marketplaces usually have good category labels for best-sellers, but the products that need categorising most are
long-tail items and new listings. We classify Amazon products from their title, description and co-purchase links.

The official split follows sales rank. In the full OGB graph, training products are the top 8%, validation the next
2% and test the rest, and this subset keeps those assignments. The test set is therefore the long tail, and its
category mix differs from training (label shift).

We work in four stages. Each one adds a source of information and is compared with the stage before.

| Stage | Question | Model in the report | Test acc | Test macro-F1 | Cold-start acc |
|---|---|---|---|---|---|
| 1. GNNs on OGB features | Does the co-purchase graph help? | GCN | 0.681 | 0.283 | 0.512 |
| 2. Text-only classifiers | How much do text and an LLM add? | fine-tuned gte-large + MLP | 0.794 | 0.379 | 0.794 |
| 3. GNNs on text embeddings | Does the graph still help on strong text? | fine-tuned gte-large + NAGphormer | 0.817 | 0.409 | 0.802 |
| 4. Post-hoc fusion | Can LLM evidence correct the category mix? | Stage 3 + LLM fusion + PPR prior + BP | 0.842 | 0.437 | 0.831 |

All numbers are means over 3 seeds; with gte-large, 3 encoder seeds × 3 GNN seeds. Cold start removes every edge
that touches a test product at inference time.

- Stage 1 trains all GNNs with one shared setup. With per-model tuning GCN reaches about 0.71
  (`results/1_gnn_ogb/03_tuned_baselines/report_table.csv`), and the conclusions stay the same.
- Stages 2 and 3 use the first three fine-tuned gte-large encoders (`exp/g-arch/table_enc.md`). Stage 4 uses three
  encoders fine-tuned again and saved for release. With those, NAGphormer alone gets 0.814 / 0.398 / 0.799 cold
  (`results/4_fusion/ablation_extra_summary.csv`).

Stage 4 puts together standard methods:

- Bayesian fusion of an LLM's zero-shot answers through its confusion matrix;
- a label-shift prior spread over the graph with personalised PageRank (PPR);
- one round of belief propagation (BP). Validation kept it only on "sibling" edges (products bought with the same
  product) and turned it off on co-purchase and text-kNN edges. It adds about 0.05 points.

See [`docs/method.md`](docs/method.md) for the method and ablations.

## Evaluation protocol

- We keep the official split and never use test labels for training or tuning.
- Epochs, temperatures, which LLM answers to fuse, the PPR setting and the BP weights are chosen on validation
  accuracy.
- Two Stage 4 settings are not tuned. The LLM weight and the prior strength λ are fixed at 1, the value Bayes' rule
  gives. Validation prefers smaller values because it has the training category mix.
- We had seen test results during the error analysis behind that choice. The evidence that 1 works better than the
  validation choice therefore comes from test.

## Dataset

We use the ogbn-products subset released with [TAPE](https://github.com/XiaoxinHe/TAPE) (commit `d9881f7`):

- 54,025 Amazon products, each with a title and a description
- 72,319 undirected co-purchase edges
- 47 category ids, of which 42 occur
- the official train / validation / test split (14,708 / 1,572 / 37,745 products)

| File | Content |
|---|---|
| `data/TAPE/ogbn-products_subset.csv` | Product text: `uid` (ASIN), `nid`, `title`, `content` |
| `data/TAPE/ogbn-products_subset.pt` | Graph, labels and split masks (PyG `Data` object); downloaded, not in git |
| `data/TAPE/labelidx2productcategory.csv.gz` | Category id to category name |
| `data/TAPE/download_manifest.json` | Source URL, size and SHA-256 of each file |

Row *i* of the CSV corresponds to node *i* of the graph.

## Installation

The code runs in the CS5284 course environment `gnn_course` (Python 3.11, PyTorch 2.1.2, PyTorch Geometric 2.5.1,
DGL 1.0.2). Install the additional packages into it:

```bash
conda activate gnn_course
pip install -r requirements.txt
```

The local LLM answers (`scripts/llm_zero_shot.py`) need a separate SGLang environment and a GPU. Fine-tuning a text
encoder (`scripts/finetune_lm.py`, `notebooks/2_text_only/tape_lm_colab.ipynb`) also needs a GPU. Everything else
runs on CPU.

## Downloads

Large files go to `outputs/`, which is not in git. Downloaded dataset files and TAPE's GPT files are checked against
SHA-256 values kept in the repository.

| What | How |
|---|---|
| Graph file `ogbn-products_subset.pt` | The EDA notebook or the Stage 4 notebook downloads it from TAPE (`data/TAPE/download_manifest.json`) |
| TAPE's GPT-3.5 answers, parsed (`outputs/tape_gpt_preds/ogbn-products.csv`) | The Stage 4 notebook downloads it; source in `results/shared/tape_gpt_preds_manifest.json` |
| TAPE's GPT-3.5 explanations (`outputs/tape_responses/ogbn-products.zip`) | Download from the Google Drive link in `results/shared/tape_gpt_responses_manifest.json` |
| Released checkpoints, embeddings and LLM answers (3.4 GB) | `python scripts/download_artifacts.py` (public Google Drive zip, no login) |

## Reproducing each stage

The notebooks keep their saved outputs, so you can read them without running anything. Cells that only run on
Google Colab (DeBERTa fine-tuning and the optional setup cells of the Stage 1 notebooks) are skipped elsewhere, and
their saved outputs show Colab paths.

| Stage | Notebook | Scripts | Results |
|---|---|---|---|
| EDA | `notebooks/0_eda/data_eda.ipynb` | `report/scripts/data_insights.py` | `notebooks/0_eda/eda_results/`, `report/tables/` |
| 1. GNNs on OGB features | `notebooks/1_gnn_ogb/gnn_experiments.ipynb` (GCN, GraphSAGE, GAT), `gt_experiments.ipynb` (Graph Transformer), `tuned_baselines.ipynb` (per-model tuning) | none: the training code is in the notebooks, the metrics in `src/gnn_evaluation.py` | `results/1_gnn_ogb/` |
| 2. Text-only classifiers | `notebooks/2_text_only/text_only_analysis.ipynb`; `tape_lm_colab.ipynb` fine-tunes DeBERTa on Colab | `run_text_only.py`, `run_text_transformer.py`, `run_gpt_pred.py`, `finetune_lm.py` (gte-large), `encode_original.py`, `encode_tape.py`, `run_tape.py`, `export_tape_lm_input.py`, `verify_manifest.py`, `analyze_truncation.py` | `results/2_text_only/`; the gte-large + MLP row is in `exp/g-arch/table_enc.md` |
| 3. GNNs on text embeddings | none | `run_baselines.py` (MiniLM + MLP, GCN, GraphSAGE), `run_graph_transformer.py`, `run_tape_repro.py` (TAPE reproduction with DeBERTa), `finetune_lm.py` (gte-large), `export_probs.py` (GNN backbones on gte-large, incl. NAGphormer) | `results/3_gnn_text/`, `exp/g-arch/` (backbone table), `exp/graph/` (text-kNN edges) |
| 4. Post-hoc fusion | `notebooks/4_fusion/final_pipeline.ipynb` (end to end, from the released checkpoints) | `run_fusion.py` (fusion + BP); `run_final.py` (adds the PPR prior and sibling edges; the final pipeline); `make_final_notebook.py`. `run_final_eval.py` and `run_final_checks.py` compare fusion rules over 33 base models in one planned test run, with weights fixed on train-only holdouts (`make_holdouts.py`, `run_holdout.py`) | `results/4_fusion/` (`final_*@*` and `final_{runs,summary,selected}.csv` from `run_final.py`; `final_{val,test}_*.csv` from `run_final_eval.py`), `exp/main/` (ablations); `run_fusion.py` writes `results/3_gnn_text/fusion_*`, `make_holdouts.py` writes `results/shared/holdouts.json` |

The final pipeline for one encoder seed, from scratch (repeat for seeds 0, 1, 2):

```bash
python scripts/encode_original.py                                  # MiniLM features for text-kNN edges (40 min CPU)
python scripts/finetune_lm.py --seed 0 --tag=-r --save-ckpt        # fine-tuned gte-large (GPU, about 20 min)
python scripts/export_probs.py NAGphormer --emb ft-gte-large-r_s0 --save-ckpt   # 3 GNN seeds, normal and cold
python scripts/run_final.py NAGphormer@ft-gte-large-r_s0           # fusion + PPR prior + BP, chosen on val
```

After `python scripts/download_artifacts.py`, skip `finetune_lm.py` and run `export_probs.py` with `--load-ckpt`
instead of `--save-ckpt` to reuse the released weights.

The figures and tables of the report are made by `report/figures/main/*.py` and `report/scripts/*.py`. Some of them
read intermediate predictions in `outputs/` that the scripts above write.

## Repository structure

```
data/TAPE/            dataset files and their download manifest
notebooks/
    0_eda/            data exploration
    1_gnn_ogb/        Stage 1 notebooks
    2_text_only/      Stage 2 notebooks
    4_fusion/         Stage 4 end-to-end notebook
src/                  shared code
    data.py           dataset loading with SHA-256 checks
    text_clean.py     text cleaning
    encode.py         sentence embeddings (frozen encoder)
    models.py         MLP, GCN, GraphSAGE, TAPE's network, Graph Transformer, NAGphormer and text Transformer
    positional.py     Laplacian positional encodings for the Graph Transformer
    train.py          training and evaluation
    train_text.py     tokenisation and mini-batch training for the text Transformer
    gnn_evaluation.py per-category metrics and seed summaries for the Stage 1 notebooks
    label_shift.py    LLM confusion matrices, prior estimation, Bayesian fusion, PPR local prior, belief propagation
    arms.py           feature sets of the Stage 2 and 3 runs
    gpt_parse.py      parsing LLM answers into category ids
    improve.py        fusion rules compared in the final evaluation
    shift.py          test-like holdouts used to fix the fusion weights
    stats.py          paired significance tests
scripts/              entry points that run the experiments (see the table above)
results/
    shared/           embedding manifests, download manifests, holdout definitions
    1_gnn_ogb/  2_text_only/  3_gnn_text/  4_fusion/   metrics and predictions per stage
exp/                  exploration code and tables cited in the report (g-arch, graph, g-localprior, g-restore, main)
report/               scripts, data and figures of the report
docs/
    method.md         the Stage 4 method, results and ablations
    research_log.md   log of the directions we tried, in order
tests/                unit tests for src/gnn_evaluation.py
```

## References

- X. He, X. Bresson, T. Laurent, A. Perold, Y. LeCun, B. Hooi. Harnessing Explanations: LLM-to-LM Interpreter for
  Enhanced Text-Attributed Graph Representation Learning. ICLR 2024.
- W. Hu et al. Open Graph Benchmark: Datasets for Machine Learning on Graphs. NeurIPS 2020.
- V. P. Dwivedi and X. Bresson. A Generalization of Transformer Networks to Graphs. AAAI Workshop on
  Deep Learning on Graphs, 2021.
- T. Kipf and M. Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
- W. Hamilton, R. Ying, J. Leskovec. Inductive Representation Learning on Large Graphs. NeurIPS 2017.
- P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. Graph Attention Networks. ICLR 2018.
- P. He, X. Liu, J. Gao, W. Chen. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR 2021.
- Q. Huang, H. He, A. Singh, S. Lim, A. Benson. Combining Label Propagation and Simple Models Out-performs Graph Neural
  Networks. ICLR 2021.
- Z. Lipton, Y.-X. Wang, A. Smola. Detecting and Correcting for Label Shift with Black Box Predictors. ICML 2018.
- A. Menon et al. Long-tail Learning via Logit Adjustment. ICLR 2021.
- J. Chen, K. Gao, G. Li, K. He. NAGphormer: A Tokenized Graph Transformer for Node Classification in Large Graphs.
  ICLR 2023.
- J. Gasteiger, A. Bojchevski, S. Günnemann. Predict then Propagate: Graph Neural Networks meet Personalized PageRank.
  ICLR 2019.
- M. Saerens, P. Latinne, C. Decaestecker. Adjusting the Outputs of a Classifier to New a Priori Probabilities: A
  Simple Procedure. Neural Computation, 2002.
- A. P. Dawid and A. M. Skene. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. 1979.
