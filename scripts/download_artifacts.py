"""Download the released artefacts (final pipeline: embeddings, GNN checkpoints, LLM answers and the three fine-tuned
gte-large encoders; TAPE reproduction: the fine-tuned DeBERTa embeddings and predictions in outputs/tape_lm/) from a
public Google Drive zip, and unpack them into outputs/.

Usage: python scripts/download_artifacts.py              # download (about 3.4 GB) and unpack
       python scripts/download_artifacts.py --zip PATH   # unpack a zip downloaded by hand from DRIVE_URL

Only entries under outputs/ are unpacked; anything else in the zip is skipped. Existing files are kept.
No login is needed. If Google Drive refuses the download (for example its daily quota), download the zip from
DRIVE_URL in a browser and pass it with --zip.
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILE_ID = "1s0HD0KxibP5zkJTaH65tGKI--BGy4_bV"
DRIVE_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
DIRECT_URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
# what the notebook needs; the encoders (outputs/ckpt/ft-gte-large-r_s*) are only used by its optional demo cell
NEEDED = ([f"outputs/embeddings/ft-gte-large-r_s{s}.npy" for s in range(3)]
          + ["outputs/embeddings/original__all-MiniLM-L6-v2.npy", "outputs/llm/qwen35_9b.npy",
             "outputs/llm_answers/qwen35_9b_knowndefs.npy", "outputs/llm_answers/qwen35_9b_rareremap.npy"]
          + [f"outputs/ckpt/gnn/NAGphormer@ft-gte-large-r_s{e}_s{s}.pt" for e in range(3) for s in range(3)])


def download(path):
    req = urllib.request.Request(DIRECT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        if r.headers.get_content_type() == "text/html":
            raise SystemExit(f"Google Drive returned a web page instead of the zip. Download it from {DRIVE_URL} "
                             "in a browser and rerun with --zip PATH.")
        total = int(r.headers.get("Content-Length", 0))
        done, step = 0, 256 << 20
        with open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % step < len(chunk):
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)


def unpack(path):
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            # skip folders, macOS metadata and hidden paths such as .git/ or .cache/ (their copies of outputs/ too)
            if (info.is_dir() or "outputs" not in parts or "__MACOSX" in parts
                    or any(p.startswith(".") for p in parts)):
                continue
            rel = "/".join(parts[parts.index("outputs"):])
            dest = REPO / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 16 << 20)
            n += 1
    return n


ap = argparse.ArgumentParser()
ap.add_argument("--zip", type=Path, help="a zip already downloaded from DRIVE_URL")
args = ap.parse_args()

zip_path = args.zip
if zip_path is None:
    zip_path = REPO / "outputs/_artifacts.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", DRIVE_URL, flush=True)
    download(zip_path)
print("unpacked", unpack(zip_path), "files into", REPO / "outputs")
if args.zip is None:
    zip_path.unlink()
missing = [f for f in NEEDED if not (REPO / f).exists()]
if missing:
    raise SystemExit("missing after unpacking: " + ", ".join(missing))
print("all files the notebook needs are in place")
