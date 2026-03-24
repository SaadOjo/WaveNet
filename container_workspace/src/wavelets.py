from __future__ import annotations

import numpy as np
import pywt
import matplotlib.cm as cm


def cwt_transform(signal, wavelet: str = "morl", scales=None):
    if scales is None:
        scales = np.arange(1, 21)
    signal = np.array(signal)
    coefs, _ = pywt.cwt(signal, scales, wavelet)
    return coefs


def normalize_image(image: np.ndarray, mode: str = "none", global_stats: dict | None = None) -> np.ndarray:
    if mode == "std" and global_stats is not None:
        image = (image - global_stats["mean"]) / (global_stats["std"] + 1e-8)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image
    if mode == "minmax" and global_stats is not None:
        return (image - global_stats["min"]) / (global_stats["max"] - global_stats["min"] + 1e-8)

    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min + 1e-8)


def compute_global_stats(images: list[np.ndarray], mode: str) -> dict | None:
    if mode == "none":
        return None
    stacked = np.asarray(images)
    if mode == "std":
        return {"mean": float(stacked.mean()), "std": float(stacked.std())}
    if mode == "minmax":
        return {"min": float(stacked.min()), "max": float(stacked.max())}
    raise ValueError(f"Unsupported normalization mode: {mode}")


def apply_jet_colormap(image: np.ndarray) -> np.ndarray:
    cmap = cm.get_cmap("jet")
    return cmap(image)[:, :, :3]
