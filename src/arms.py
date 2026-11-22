"""Features and models of every trained arm, so later scripts rebuild them the same way from saved weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_tape_explanations
from src.gpt_parse import category_patterns, load_tape_gpt_preds, ranked_categories
from src.models import GCN, MLP, SAGE, TapeNet

REPO = Path(__file__).resolve().parent.parent
NUM_CLASSES = 47
HIDDEN = 256
CLASSES = {"MLP": MLP, "GCN": GCN, "SAGE": SAGE}
BASE_WEIGHTS = REPO / "outputs" / "weights"  # scripts/run_baselines.py, run_tape.py, run_gpt_pred.py
REPRO_WEIGHTS = REPO / "outputs" / "tape_repro"  # scripts/run_tape_repro.py

# arm: (model, feature set). K and L are left out: nothing reuses them.
BASE_ARMS = {
    "A": ("MLP", "orig"), "B": ("GCN", "orig"), "SAGE": ("SAGE", "orig"),
    "G": ("MLP", "orig+tape"), "H": ("GCN", "orig+tape"),
    "P_MLP": ("MLP", "orig+gpt"), "P_GCN": ("GCN", "orig+gpt"),
}
REPRO_FEATURES = ["orig", "expl", "pred"]
REPRO_TOP_K = {"tape": 5, "ours": 3}


def checked_load(path, manifest_path):
    """np.load(path) after checking its SHA-256 against the manifest."""
    meta = json.loads(Path(manifest_path).read_text())
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{Path(path).name} does not match {manifest_path}")
    return np.load(path)


def gpt_rank(num_nodes, k=3):
    """GPT's ranked categories from our parse (src/gpt_parse.py): (num_nodes, k), -1 = none."""
    patterns = category_patterns()
    return torch.tensor([ranked_categories(t, patterns, k) for t in load_tape_explanations(num_nodes)])


def rank_one_hot(rank):
    """One-hot of each rank, concatenated: (num_nodes, k * NUM_CLASSES). A missing answer is all zeros."""
    return torch.cat([F.one_hot(rank[:, k].clamp(min=0), NUM_CLASSES) * (rank[:, k:k + 1] >= 0)
                      for k in range(rank.size(1))], dim=1).float()


class Features:
    """Feature sets by name, each built once on first use."""

    def __init__(self, num_nodes, lm_seed=0):
        self.num_nodes, self.lm_seed, self.cache = num_nodes, lm_seed, {}

    def _minilm(self, name):
        return torch.from_numpy(checked_load(REPO / "outputs" / "embeddings" / f"{name}.npy",
                                             REPO / "results" / "shared" / "embeddings" / f"{name}.json"))

    def _lm(self, variant):
        name = f"tape_lm_{variant}_deberta-base_seed{self.lm_seed}"
        return torch.from_numpy(checked_load(REPO / "outputs" / "tape_lm" / f"{name}.emb.npy",
                                             REPO / "results" / "shared" / "embeddings" / f"{name}.json").astype(np.float32))

    def _build(self, name):
        if name == "orig":
            return self._minilm("original__all-MiniLM-L6-v2")
        if name == "orig+tape":
            return torch.cat([self["orig"], self._minilm("tape__all-MiniLM-L6-v2")], dim=1)
        if name == "orig+gpt":
            return torch.cat([self["orig"], self["gpt3"]], dim=1)
        if name == "gpt3":
            return rank_one_hot(self["rank"][:, :3])
        if name == "rank":
            return gpt_rank(self.num_nodes, 5)
        if name == "gpt5_tape":
            meta = json.loads((REPO / "results" / "shared" / "tape_gpt_preds_manifest.json").read_text())
            path = REPO / meta["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"{path} does not match results/shared/tape_gpt_preds_manifest.json")
            return load_tape_gpt_preds(path, self.num_nodes, 5) + 1
        if name in ("lm_orig", "lm_expl"):
            return self._lm(name[3:])
        raise KeyError(name)

    def __getitem__(self, name):
        if name not in self.cache:
            self.cache[name] = self._build(name)
        return self.cache[name]


def base_model(arm, features):
    """A fresh model for a base arm and its features."""
    kind, feature = BASE_ARMS[arm]
    x = features[feature]
    return CLASSES[kind](x.size(1), HIDDEN, NUM_CLASSES), x


def load_base(arm, seed, features):
    """A base arm's trained model (eval mode) and its features."""
    model, x = base_model(arm, features)
    model.load_state_dict(torch.load(BASE_WEIGHTS / f"{arm}_s{seed}.pt", weights_only=True))
    return model.eval(), x


def repro_x(config, feature, features):
    if feature != "pred":
        return features[f"lm_{feature}"]
    return features["gpt5_tape"] if config == "tape" else features["gpt3"]


def load_repro(config, feature, kind, seed, features):
    """A model from scripts/run_tape_repro.py (eval mode) and its features, for the DeBERTa seed in features.

    GPT's answers do not depend on the DeBERTa seed, so the pred models exist only under DeBERTa seed 0.
    """
    lm_seed = 0 if feature == "pred" else features.lm_seed
    x = repro_x(config, feature, features)
    if config == "ours":
        model = CLASSES[kind](x.size(1), HIDDEN, NUM_CLASSES)
    else:
        use_pred = feature == "pred"
        model = TapeNet(kind, 128 * REPRO_TOP_K["tape"] if use_pred else x.size(1), 128, NUM_CLASSES,
                        use_pred=use_pred)
    path = REPRO_WEIGHTS / f"{config}_lm{lm_seed}_{feature}_{kind}_s{seed}.pt"
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval(), x
