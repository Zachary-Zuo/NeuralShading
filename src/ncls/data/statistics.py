from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplicaMoments:
    """一个独立随机流的总体均值、总体方差和逐 query group 样本数。"""

    mean: np.ndarray
    variance: np.ndarray
    sample_count: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        variance = np.asarray(self.variance, dtype=np.float64)
        count = np.asarray(self.sample_count, dtype=np.uint64)
        if mean.shape != variance.shape:
            raise ValueError("replica mean and variance must have the same shape")
        if mean.ndim < 2 or mean.shape[-1] != 3:
            raise ValueError("replica responses must end in an RGB axis")
        expected_count_shape = mean.shape[:1]
        if count.ndim == 0 and mean.shape[0] == 1:
            count = count.reshape(1)
        if count.shape != expected_count_shape:
            raise ValueError("sample_count must contain one value per leading query group")
        if np.any(count == 0):
            raise ValueError("sample_count must be positive")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("replica statistics must be finite")
        if np.any(variance < 0.0):
            raise ValueError("replica variance must be nonnegative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "sample_count", count)


@dataclass(frozen=True)
class CombinedMoments:
    mean: np.ndarray
    variance: np.ndarray
    sample_count: np.ndarray
    standard_error: np.ndarray


def _broadcast_count(count: np.ndarray, ndim: int) -> np.ndarray:
    return count.reshape((len(count),) + (1,) * (ndim - 1)).astype(np.float64)


def combine_replica_moments(a: ReplicaMoments, b: ReplicaMoments) -> CombinedMoments:
    """精确合并两个独立流的总体矩，不用二阶矩相减。"""

    if a.mean.shape != b.mean.shape:
        raise ValueError("replicas must have matching response shapes")
    count_a = _broadcast_count(a.sample_count, a.mean.ndim)
    count_b = _broadcast_count(b.sample_count, b.mean.ndim)
    total_count = count_a + count_b
    delta = b.mean - a.mean
    mean = a.mean + delta * (count_b / total_count)
    m2 = a.variance * count_a + b.variance * count_b
    m2 += delta * delta * count_a * count_b / total_count
    variance = np.maximum(m2 / total_count, 0.0)
    standard_error = np.sqrt(variance / total_count)
    return CombinedMoments(
        mean,
        variance,
        a.sample_count + b.sample_count,
        standard_error,
    )
