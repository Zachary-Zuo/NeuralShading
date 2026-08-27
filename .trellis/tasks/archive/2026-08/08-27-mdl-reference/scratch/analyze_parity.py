import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--asset-id", action="append")
arguments = parser.parse_args()
root = arguments.root
asset_ids = tuple(
    arguments.asset_id
    or ("carpaint-shifting-flakes", "copper-antique-brushed-patinated")
)
for asset_id in asset_ids:
    formal = np.load(root / asset_id / "formal.npz")
    oracle = np.load(root / asset_id / "oracle/result.npz")
    formal_value, oracle_value = formal["value"], oracle["value"]
    formal_pdf, oracle_pdf = formal["pdf"], oracle["pdf"]
    print(f"\n{asset_id} {formal_value.shape}")
    for surface in range(formal_value.shape[0]):
        for view in range(formal_value.shape[1]):
            for light in range(formal_value.shape[2]):
                lhs = formal_value[surface, view, light]
                rhs = oracle_value[surface, view, light]
                denominator = max(float(np.abs(lhs).sum()), float(np.abs(rhs).sum()), 3e-5)
                relative = float(np.abs(lhs - rhs).sum()) / denominator
                if relative < 0.001 and abs(float(formal_pdf[surface, view, light]) - float(oracle_pdf[surface, view, light])) < 0.0002:
                    continue
                print(
                    surface,
                    view,
                    light,
                    "rel",
                    relative,
                    "formal",
                    lhs.tolist(),
                    "oracle",
                    rhs.tolist(),
                    "pdf",
                    float(formal_pdf[surface, view, light]),
                    float(oracle_pdf[surface, view, light]),
                )
