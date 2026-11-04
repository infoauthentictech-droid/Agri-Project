"""Decode and transform image batches on the GPU.

Only reading the file bytes stays on the CPU, because the disk cannot be read any other way.
JPEG decoding uses nvJPEG and every transform runs on GPU tensors, so training never waits
on CPU image work. Non-JPEG files and JPEGs nvJPEG rejects decode on the CPU and move to the
GPU at once; ``cpu_decoded`` counts them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, List, Sequence, Tuple

import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_image, decode_jpeg, read_file
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def decode_file(path: Path, device: str = "cuda") -> Tuple[torch.Tensor, bool]:
    """Return a CxHxW RGB uint8 image on the GPU and whether nvJPEG decoded it.

    EXIF-rotated photos must honour their orientation, which only the CPU decoder can do.
    """
    data = read_file(str(path))
    with Image.open(path) as probe:
        rotated = probe.getexif().get(274, 1) not in (0, 1)
    if not rotated:
        try:
            return decode_jpeg(data, mode=ImageReadMode.RGB, device=device), True
        except RuntimeError:
            pass
    return decode_image(data, mode=ImageReadMode.RGB, apply_exif_orientation=True).to(device), False


def classifier_transform(image_size: int, train: bool) -> v2.Compose:
    """The classifier's augmentation pipeline, run on GPU tensors."""
    steps: List = [v2.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [
            v2.Resize((int(image_size * 1.1), int(image_size * 1.1)), antialias=True),
            v2.RandomCrop(image_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation(degrees=30),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ]
    else:
        steps += [
            v2.Resize((image_size, image_size), antialias=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return v2.Compose(steps)


def cyclegan_transform(image_size: int) -> v2.Compose:
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((image_size, image_size), antialias=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class GpuBatchLoader:
    """Yield (images, labels) batches already on the GPU from ImageFolder-style samples."""

    def __init__(self, samples: Sequence[Tuple[str, int]], batch_size: int, transform: Callable,
                 shuffle: bool, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. Image loading is GPU-only.")
        self.samples = list(samples)
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.device = device
        self.cpu_decoded = 0

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def _decode(self, paths: Sequence[str]) -> List[torch.Tensor]:
        try:
            return decode_jpeg([read_file(p) for p in paths], mode=ImageReadMode.RGB, device=self.device)
        except RuntimeError:
            # One unsupported file fails the whole batch, so retry each file alone.
            images = []
            for path in paths:
                image, on_gpu = decode_file(Path(path), self.device)
                self.cpu_decoded += not on_gpu
                images.append(image)
            return images

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        order = torch.randperm(len(self.samples)).tolist() if self.shuffle else range(len(self.samples))
        order = list(order)
        for start in range(0, len(order), self.batch_size):
            batch = [self.samples[i] for i in order[start:start + self.batch_size]]
            images = self._decode([path for path, _ in batch])
            inputs = torch.stack([self.transform(image) for image in images])
            yield inputs, torch.tensor([label for _, label in batch], device=self.device)
