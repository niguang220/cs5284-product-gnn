"""Load TAPE's ogbn-products subset: graph, labels and split from the .pt file, text from the CSV."""
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "TAPE"
TAPE_ZIP = Path(__file__).resolve().parent.parent / "outputs" / "tape_responses" / "ogbn-products.zip"
TAPE_MANIFEST = Path(__file__).resolve().parent.parent / "results" / "shared" / "tape_gpt_responses_manifest.json"


def _check_graph_file(path, data_dir):
    # torch.load unpickles arbitrary objects, so only load the exact file pinned in the manifest.
    manifest = json.loads((data_dir / "download_manifest.json").read_text())
    expected = next(item["sha256"] for item in manifest if item["artifact"] == "graph")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{path.name} does not match download_manifest.json; re-download it.")


def load_products(data_dir=DATA_DIR):
    """Return (graph, text).

    graph: PyG Data with an undirected edge_index, labels y of shape [N], and train/val/test masks.
    text:  DataFrame with uid, nid, title, content; row i is node i.
    """
    data_dir = Path(data_dir)
    graph_path = data_dir / "ogbn-products_subset.pt"
    _check_graph_file(graph_path, data_dir)
    raw = torch.load(graph_path)
    text = pd.read_csv(data_dir / "ogbn-products_subset.csv")

    # Row i of the CSV must be node i of the graph.
    assert len(text) == raw.num_nodes
    assert (text["uid"].values == raw.n_asin).all()
    assert (text["nid"].values == raw.n_id.numpy()).all()

    # Most co-purchase edges are stored in one direction only; GCN needs them both ways.
    row, col, _ = raw.adj_t.coo()
    edge_index = to_undirected(torch.stack([row, col]), num_nodes=raw.num_nodes)

    graph = Data(
        edge_index=edge_index,
        y=raw.y.view(-1),
        train_mask=raw.train_mask,
        val_mask=raw.val_mask,
        test_mask=raw.test_mask,
        num_nodes=raw.num_nodes,
    )
    return graph, text


def load_tape_explanations(num_nodes, zip_path=TAPE_ZIP, manifest_path=TAPE_MANIFEST):
    """Return TAPE's GPT explanation of every node as a list of str; item i is node i.

    Read straight from the downloaded zip, which must match its manifest in results/.
    """
    expected = json.loads(Path(manifest_path).read_text())[0]["sha256"]
    if hashlib.sha256(Path(zip_path).read_bytes()).hexdigest() != expected:
        raise ValueError(f"{Path(zip_path).name} does not match {Path(manifest_path).name}; re-download it.")
    with zipfile.ZipFile(zip_path) as z:
        return [json.loads(z.read(f"ogbn-products/{i}.json"))["choices"][0]["message"]["content"]
                for i in range(num_nodes)]
