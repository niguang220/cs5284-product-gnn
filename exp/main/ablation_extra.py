"""Two controls for the final pipeline's ablation, under the scripts/run_final.py protocol.

Usage: python exp/main/ablation_extra.py [BASE ...]   (default: NAGphormer MLP)
  needs outputs/probs/{NAGphormer,MLP}@ft-gte-large-r_s{0,1,2}_{normal,cold}_s{0,1,2}.npy
  (MLP: scripts/export_probs.py MLP --emb ft-gte-large-r_s{e}; an MLP has no cold file, its probabilities do not
  depend on the edges, so the normal file is used in both settings).

Per base model, encoder seed and setting:
  ppr     the run_final.py grid (LLM combination, PPR (a, iters), β, γ, g2), chosen on val accuracy (mean over the
          GNN seeds); the steps of that configuration switched on in turn, as in notebooks/4_fusion/final_pipeline.ipynb:
          base -> + LLM fusion -> + PPR local prior (λ = 1) -> + BP.
  global  the same grid with the PPR prior replaced by one global prior π̂ (Saerens et al. 2002 EM,
          src/label_shift.em_prior, on the fused posteriors of all val + test nodes; no graph), added as
          λ · log π̂ / π_tr with λ = 1. Chosen on val over (LLM combination, β, γ, g2); rows: + global prior (BP off)
          and + global prior + BP.
  global-mix  as global, but π̂ is mixed 0.9 / 0.1 with π_tr, as ppr_log_prior mixes its diffused prior
          (the closest global counterpart of the PPR prior).
  noprior the same grid without any prior (+ fusion + BP), for reference.
λ = 1 and LLM weight 1 are fixed as in run_final.py. Test labels are only read for reporting.
Writes results/4_fusion/ablation_extra_{runs,summary}.csv and results/4_fusion/ablation_extra_meta.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
torch.set_num_threads(8)
from scripts import run_final as rf  # noqa: E402  builds the LLM likelihoods and BP messages once
from src.label_shift import K, em_prior  # noqa: E402
from src.train import masked_metrics  # noqa: E402

y, va, te, tr = rf.y, rf.va, rf.te, rf.tr
yt = torch.from_numpy(y)
# Same train prior as ppr_log_prior, so the global and the local prior differ only in π̂.
PI_TR = np.maximum(np.bincount(y[tr], minlength=K) / tr.sum(), 1e-4)
PI_TR /= PI_TR.sum()
BASES = sys.argv[1:] or ["NAGphormer", "MLP"]
ENCS = [0, 1, 2]
SEEDS = [0, 1, 2]


def global_prior(fused):
    """π̂ = Saerens EM on the fused posteriors of the unlabelled (val + test) nodes."""
    q = np.exp(fused[~tr])
    pi = np.maximum(em_prior(q / q.sum(1, keepdims=True), PI_TR), 1e-12)  # an underflowing class would give log 0
    return pi / pi.sum()


GLOBALS = {"global": lambda pi: np.log(pi / PI_TR), "global-mix": lambda pi: np.log((0.9 * pi + 0.1 * PI_TR) / PI_TR)}


def load(base, enc, setting, seed):
    s = "normal" if base == "MLP" else setting  # the MLP ignores edges
    return np.load(REPO / f"outputs/probs/{base}@ft-gte-large-r_s{enc}_{s}_s{seed}.npy").astype(np.float64)


def val_acc(zv):
    return (zv.argmax(1) == y[va]).mean()


def select(rows, key):
    df = pd.DataFrame(rows)
    return dict(zip(key, df.groupby(key)["val_acc"].mean().idxmax()))


def metrics(z):
    pred = torch.from_numpy(z.argmax(1))
    v, t = masked_metrics(yt, pred, torch.from_numpy(va)), masked_metrics(yt, pred, torch.from_numpy(te))
    return {"val_acc": v["acc"], "test_acc": t["acc"], "test_f1": t["macro_f1"]}


runs, configs = [], []
PI_TE = np.bincount(y[te], minlength=K) / te.sum()  # reporting only
for base in BASES:
    for enc in ENCS:
        tag = f"{base}@ft-gte-large-r_s{enc}"
        for setting in ["normal", "cold"]:
            LP, FUSED, PI = {}, {}, {}
            grids = {g: [] for g in ["ppr", *GLOBALS, "noprior"]}
            for seed in SEEDS:
                p = load(base, enc, setting, seed)
                LP[seed] = lp = rf.log_softmax(np.log(np.clip(p, 1e-12, 1)) / rf.fit_temperature(p, y, va))
                for combo in rf.combos:
                    FUSED[seed, combo] = fused = rf.log_softmax(lp + rf.LLMLOG[combo])
                    PI[seed, combo] = global_prior(fused)
                    # The grid only needs val accuracy, so it works on the val rows.
                    mv, kv, sv = rf.MSG[(setting, combo)][va], rf.KMSG[combo][va], rf.SMSG[(setting, combo)][va]
                    c = "+".join(combo)

                    def grid(zv, **extra):  # additions in run_final.py's order, so rounding (and ties) match it
                        return [{"llms": c, **extra, "beta": b, "gamma": g, "g2": g2,
                                 "val_acc": val_acc((zv + b * mv + g * kv) + g2 * sv)}
                                for b in rf.BETAS for g in rf.GAMMAS for g2 in rf.G2S]
                    for ppr in rf.PPR:
                        z0 = fused + rf.LAM * rf.ppr_log_prior(fused, rf.ppr_graph[setting], y, tr, *ppr)
                        grids["ppr"] += grid(z0[va], a=ppr[0], iters=ppr[1])
                    for name, f in GLOBALS.items():
                        grids[name] += grid((fused + rf.LAM * f(PI[seed, combo]))[va])
                    grids["noprior"] += grid(fused[va])
            cfg = {n: select(g, [k for k in g[0] if k != "val_acc"]) for n, g in grids.items()}
            for n, c in cfg.items():
                row = {"model": tag, "setting": setting, "grid": n, **c}
                if n in GLOBALS:
                    pi = np.mean([PI[s, tuple(c["llms"].split("+"))] for s in SEEDS], 0)
                    row["pi_hat_tv_to_train"] = 0.5 * np.abs(pi - PI_TR).sum()
                    row["pi_hat_tv_to_test"] = 0.5 * np.abs(pi - PI_TE).sum()
                configs.append(row)

            def with_bp(z, c):  # same order of additions as run_final.posterior
                k = tuple(c["llms"].split("+"))
                return z + c["beta"] * rf.MSG[(setting, k)] + c["gamma"] * rf.KMSG[k] + c["g2"] * rf.SMSG[(setting, k)]
            for seed in SEEDS:
                cp = cfg["ppr"]
                fused = FUSED[seed, tuple(cp["llms"].split("+"))]
                prior = fused + rf.LAM * rf.ppr_log_prior(fused, rf.ppr_graph[setting], y, tr, cp["a"], int(cp["iters"]))
                steps = {"base": LP[seed], "+ LLM fusion": fused, "+ PPR local prior": prior,
                         "+ BP (final)": with_bp(prior, cp)}
                for name, f in GLOBALS.items():
                    k = tuple(cfg[name]["llms"].split("+"))
                    z = FUSED[seed, k] + rf.LAM * f(PI[seed, k])
                    steps[f"+ {name} prior"], steps[f"+ {name} prior + BP"] = z, with_bp(z, cfg[name])
                cn = cfg["noprior"]
                steps["+ fusion + BP, no prior"] = with_bp(FUSED[seed, tuple(cn["llms"].split("+"))], cn)
                for step, z in steps.items():
                    runs.append({"base": base, "enc": enc, "setting": setting, "seed": seed, "step": step,
                                 **metrics(z)})
            print(base, enc, setting, cfg, flush=True)

runs = pd.DataFrame(runs)
runs.to_csv(REPO / "results/4_fusion/ablation_extra_runs.csv", index=False)
# Mean over GNN seeds per encoder, then mean ± std over the 3 encoders (as notebooks/4_fusion/final_pipeline.ipynb).
per_enc = runs.groupby(["base", "setting", "step", "enc"], sort=False)[["val_acc", "test_acc", "test_f1"]].mean()
summ = per_enc.groupby(["base", "setting", "step"], sort=False).agg(["mean", "std"]).round(4)
summ.to_csv(REPO / "results/4_fusion/ablation_extra_summary.csv")
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(summ)
(REPO / "results/4_fusion/ablation_extra_meta.json").write_text(json.dumps({
    "lambda": rf.LAM, "llm_weight": 1, "ppr": rf.PPR, "betas": rf.BETAS, "gammas": rf.GAMMAS, "g2s": rf.G2S,
    "knn": rf.KNN, "llms": list(rf.llm), "combos": ["+".join(c) for c in rf.combos],
    "global_prior": "src.label_shift.em_prior (Saerens EM) on the fused posteriors of val + test nodes, "
                    "pi_tr as in ppr_log_prior (floor 1e-4), lambda = 1; global-mix uses 0.9 pi_hat + 0.1 pi_tr",
    "selection": "val accuracy, mean over GNN seeds, per base, encoder seed and setting; steps of the selected "
                 "configuration switched on in turn",
    "mlp_probs": "scripts/export_probs.py MLP --emb ft-gte-large-r_s{0,1,2} --device cpu --save-ckpt; normal probs "
                 "used in cold start (an MLP ignores edges)",
    "selected": configs}, indent=1, default=float))
