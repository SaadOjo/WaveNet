from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.wavelets import apply_jet_colormap, compute_global_stats, cwt_transform, normalize_image


class CWTTimeSeriesDataset(Dataset):
    def __init__(
        self,
        x_df=None,
        y_list=None,
        *,
        wavelet: str = "morl",
        scales=None,
        new_size: tuple[int, int] | None = None,
        norm_type: str = "none",
        colorize: bool = False,
        raw_images=None,
        labels: list[int] | None = None,
    ):
        if scales is None:
            scales = np.arange(1, 21)

        self.wavelet = wavelet
        self.scales = [int(s) for s in scales]
        self.new_size = new_size
        self.norm_type = norm_type
        self.colorize = colorize

        if raw_images is None:
            if x_df is None or y_list is None:
                raise ValueError("Either provide x_df/y_list or cached raw_images/labels.")

            raw_images = []
            labels = [int(y) for y in y_list]
            for i in tqdm(range(len(x_df)), desc="Generating CWT"):
                ts = x_df.iloc[i].iloc[0]
                raw_images.append(cwt_transform(ts, wavelet, scales))
            raw_images = np.asarray(raw_images, dtype=np.float32)
        else:
            if labels is None:
                raise ValueError("Cached dataset requires labels.")
            if isinstance(raw_images, torch.Tensor):
                raw_images = raw_images.cpu().numpy()
            raw_images = np.asarray(raw_images, dtype=np.float32)
            labels = [int(y) for y in labels]

        self.raw_images = torch.as_tensor(raw_images, dtype=torch.float32)
        self.labels = labels
        self.global_stats = compute_global_stats(self.raw_images.numpy(), norm_type)

    @classmethod
    def from_cache(cls, payload: dict, **runtime_kwargs):
        return cls(raw_images=payload["raw_images"], labels=payload["labels"], **runtime_kwargs)

    def to_cache_payload(self) -> dict:
        return {
            "cache_format": "raw_cwt_v2",
            "raw_images": self.raw_images,
            "labels": self.labels,
            "wavelet": self.wavelet,
            "scales": self.scales,
        }

    def _transform_raw_image(self, raw_image: torch.Tensor) -> torch.Tensor:
        image = raw_image.numpy()
        if self.new_size is not None:
            image_tensor = raw_image.unsqueeze(0).unsqueeze(0)
            resize_mode = "nearest" if self.colorize else "bilinear"
            if resize_mode == "nearest":
                image = F.interpolate(image_tensor, size=self.new_size, mode=resize_mode).squeeze().numpy()
            else:
                image = (
                    F.interpolate(image_tensor, size=self.new_size, mode=resize_mode, align_corners=False)
                    .squeeze()
                    .numpy()
                )

        if self.colorize:
            image = normalize_image(image, mode=self.norm_type, global_stats=self.global_stats)
            image = apply_jet_colormap(image)
            return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if self.norm_type != "none":
            image = normalize_image(image, mode=self.norm_type, global_stats=self.global_stats)
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    def plot(self, idx: int):
        img = self[idx][0]
        plt.figure(figsize=(6, 6))
        if img.shape[0] == 1:
            plt.imshow(img.squeeze(0).numpy(), cmap="PRGn", aspect="auto")
        else:
            plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"Sample {idx}")
        plt.axis("off")
        plt.show()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._transform_raw_image(self.raw_images[idx]), self.labels[idx]


class CIFAR10GrayDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, img_size: tuple[int, int] = (20, 20)):
        self.samples = []
        self.labels = []
        transform_pipeline = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        cifar_data = datasets.CIFAR10(root=root, train=train, download=True, transform=transform_pipeline)
        for img, label in tqdm(cifar_data, desc=f"Loading CIFAR10 {'Train' if train else 'Test'}"):
            self.samples.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def _normalize_scales(scales) -> list[int]:
    if scales is None:
        return list(range(1, 21))
    return [int(s) for s in scales]


def build_cwt_cache_path(
    cache_dir: Path,
    *,
    dataset_name: str,
    split_name: str,
    wavelet: str,
    scales,
    new_size=None,
    norm_type: str = "none",
    colorize: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "cache_format": "raw_cwt_v2",
        "dataset": dataset_name,
        "split": split_name,
        "wavelet": wavelet,
        "scales": _normalize_scales(scales),
    }
    digest = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    return cache_dir / f"{dataset_name}__{split_name}__rawcwt__{digest}.pt"


def load_or_build_cwt_dataset(
    *,
    x_df,
    y_list,
    cache_dir: Path,
    dataset_name: str,
    split_name: str,
    wavelet: str = "morl",
    scales=None,
    new_size: tuple[int, int] | None = None,
    norm_type: str = "none",
    colorize: bool = False,
    rebuild_cache: bool = False,
    use_cache: bool = True,
):
    cache_path = build_cwt_cache_path(
        cache_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        wavelet=wavelet,
        scales=scales,
        new_size=new_size,
        norm_type=norm_type,
        colorize=colorize,
    )

    if use_cache and cache_path.exists() and not rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and "raw_images" in payload and "labels" in payload:
            return (
                CWTTimeSeriesDataset.from_cache(
                    payload,
                    wavelet=wavelet,
                    scales=scales,
                    new_size=new_size,
                    norm_type=norm_type,
                    colorize=colorize,
                ),
                cache_path,
                True,
            )

    dataset = CWTTimeSeriesDataset(
        x_df,
        y_list,
        wavelet=wavelet,
        scales=scales,
        new_size=new_size,
        norm_type=norm_type,
        colorize=colorize,
    )
    if use_cache:
        torch.save(dataset.to_cache_payload(), cache_path)
    return dataset, cache_path, False
