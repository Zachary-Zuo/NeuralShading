from __future__ import annotations

from pathlib import Path

import numpy as np

from ncls.learning.evaluation.sampler_correctness import (
    _quadrature,
    independent_unified_sampler_pdf,
)


ROOT = Path(".trellis/tasks/08-25-03-neural-baseline-and-candidate/scratch")
directions, _ = _quadrature()
for sampler, name in (("nvidia-diffuse-ggx9", "ggx"), ("ltc-k2", "ltc")):
    cases = np.load(ROOT / f"sampler-audit-{name}-cases.npz", allow_pickle=False)
    result = np.load(ROOT / f"sampler-audit-{name}-falcor/case-000.npz", allow_pickle=False)
    oracle = independent_unified_sampler_pdf(
        cases["prepared"][0], cases["views"][0], directions, sampler
    )
    np.testing.assert_allclose(result["queried_pdf"], oracle, rtol=2e-5, atol=1e-7)
    sampled = result["sampled"]
    metadata = result["metadata"]
    continuous = (metadata[:, 1] == 1.0) & (metadata[:, 2] == 0.0)
    np.testing.assert_allclose(
        sampled[continuous, 3], metadata[continuous, 0], rtol=2e-5, atol=1e-7
    )
    assert np.all(metadata[:, 1] == 1.0)
    print(sampler, float(np.max(np.abs(result["queried_pdf"] - oracle))))
