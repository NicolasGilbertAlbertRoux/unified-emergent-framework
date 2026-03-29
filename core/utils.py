from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_field(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    max_val = np.max(arr) if arr.size else 0.0
    return arr / max_val if max_val > 0 else arr


def smooth_and_normalize(arr: np.ndarray, sigma: float) -> np.ndarray:
    arr = normalize_field(arr)
    arr = gaussian_filter(arr, sigma=sigma)
    return normalize_field(arr)


def inverse_participation_ratio(vec: np.ndarray) -> float:
    x = np.asarray(vec, dtype=float)
    norm2 = np.sum(x ** 2)
    if norm2 <= 0:
        return float("nan")
    x = x / np.sqrt(norm2)
    return float(np.sum(x ** 4))


def effective_support(vec: np.ndarray) -> float:
    ipr = inverse_participation_ratio(vec)
    if not np.isfinite(ipr) or ipr <= 0:
        return float("nan")
    return float(1.0 / ipr)


def wave_effective_support(psi: np.ndarray) -> float:
    p = np.abs(psi) ** 2
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    ipr = np.sum(p ** 2)
    return float(1.0 / ipr) if ipr > 0 else float("nan")


def shannon_entropy(psi: np.ndarray) -> float:
    p = np.abs(psi) ** 2
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    p = np.clip(p, 1e-16, None)
    return float(-np.sum(p * np.log(p)))


def central_gradient(arr: np.ndarray, axis: int) -> np.ndarray:
    return 0.5 * (np.roll(arr, -1, axis=axis) - np.roll(arr, +1, axis=axis))


def central_laplacian(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(arr, -1, axis=axis) - 2.0 * arr + np.roll(arr, +1, axis=axis)


def hessian_tensor(arr: np.ndarray) -> np.ndarray:
    ndim = arr.ndim
    hess = np.zeros((ndim, ndim) + arr.shape, dtype=float)

    for i in range(ndim):
        hess[i, i] = central_laplacian(arr, i)

    for i in range(ndim):
        for j in range(i + 1, ndim):
            app = np.roll(np.roll(arr, -1, axis=i), -1, axis=j)
            apm = np.roll(np.roll(arr, -1, axis=i), +1, axis=j)
            amp = np.roll(np.roll(arr, +1, axis=i), -1, axis=j)
            amm = np.roll(np.roll(arr, +1, axis=i), +1, axis=j)
            value = 0.25 * (app - apm - amp + amm)
            hess[i, j] = value
            hess[j, i] = value

    return hess