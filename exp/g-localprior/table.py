"""Markdown table of the val-selected rows (summary*_selected_per_enc.csv), mean over encoder seeds, with the paired
difference to the baseline per encoder seed. Usage: python table.py [suffix ...]  (e.g. "" _ppr _emcal)."""
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).parent
sufs = sys.argv[1:] or [""]
s = pd.concat([pd.read_csv(D / f"summary{x}_selected_per_enc.csv") for x in sufs]).drop_duplicates(
    ["enc", "setting", "variant"])
base = s[s.variant == "baseline"].set_index(["enc", "setting"])
for setting in ["normal", "cold"]:
    print(f"\n{setting}:\n")
    print("| variant | val acc | test acc (mean ± std over enc seeds) | Δ acc vs baseline per enc seed (pt) | test F1 | "
          "1 hop | 2 | 3 | 4+ | Elec / Office / Tools / Auto recall |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v, g in s[s.setting == setting].groupby("variant", sort=False):
        g = g.set_index("enc").sort_index()
        b = base.xs(setting, level=1).sort_index()
        d = ((g.test_acc - b.test_acc) * 100).round(2).tolist()
        rec = " / ".join(f"{g[f'rec_{n}'].mean():.2f}" for n in ["Electronics", "Office", "Tools", "Automotive"])
        print(f"| {v} | {g.val_acc.mean():.4f} | {g.test_acc.mean():.4f} ± {g.test_acc.std():.4f} | "
              f"{', '.join(f'{x:+.2f}' for x in d)} | {g.test_f1.mean():.4f} | {g.hop1.mean():.3f} | "
              f"{g.hop2.mean():.3f} | {g.hop3.mean():.3f} | {g['hop4+'].mean():.3f} | {rec} |")
