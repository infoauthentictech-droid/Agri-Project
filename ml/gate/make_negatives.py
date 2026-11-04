"""Generate a deterministic synthetic set of NON-plant images.

These are stand-ins until real user-supplied negatives are available. They
cover the visual families that must never reach the detector: blank frames,
pavement/street scenes, people/objects/animals silhouettes, rooms, documents,
noise, gradients and graphics.

Real negatives can be dropped into ``datasets/negatives_real/`` (photos of
people, objects, animals, roads, buildings, indoor scenes) and the trainer will
pick them up automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw  # noqa: F401


def _canvas(w: int, h: int, color) -> Image.Image:
    return Image.fromarray(color) if isinstance(color, np.ndarray) \
        else Image.new("RGB", (w, h), color)


def _flat(rng, w, h):
    return _canvas(w, h, tuple(int(c) for c in rng.integers(0, 256, 3)))


def _gradient(rng, w, h, horizontal=True):
    c0 = rng.integers(0, 256, 3).astype(np.float32)
    c1 = rng.integers(0, 256, 3).astype(np.float32)
    t = np.linspace(0, 1, w if horizontal else h)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        if horizontal:
            img[:, :, c] = c0[c] * (1 - t[None, :]) + c1[c] * t[None, :]
        else:
            img[:, :, c] = c0[c] * (1 - t[:, None]) + c1[c] * t[:, None]
    return Image.fromarray(img.astype(np.uint8))


def _noise(rng, w, h):
    mode = int(rng.integers(0, 3))
    base = rng.integers(0, 256, 3)
    if mode == 0:
        arr = rng.normal(base, rng.integers(8, 60), (h, w, 3))
    elif mode == 1:
        arr = rng.integers(0, 256, (h, w, 3))
    else:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        bs = 24
        for by in range(0, h, bs):
            for bx in range(0, w, bs):
                arr[by:by + bs, bx:bx + bs] = rng.integers(0, 256, 3)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _pattern(rng, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    kind = int(rng.integers(0, 4))
    c0 = tuple(int(c) for c in rng.integers(40, 220, 3))
    c1 = tuple(int(c) for c in rng.integers(40, 220, 3))
    if kind == 0:
        s = int(rng.integers(20, 60))
        for y in range(0, h, s):
            for x in range(0, w, s):
                d.rectangle([(x, y), (x + s, y + s)],
                            fill=c0 if ((x // s + y // s) % 2 == 0) else c1)
    elif kind == 1:
        s = int(rng.integers(8, 40))
        for x in range(0, w, 2 * s):
            d.rectangle([(x, 0), (x + s, h)], fill=c1)
    elif kind == 2:
        d.rectangle([(0, 0), (w, h)], fill=c0)
        s = int(rng.integers(30, 90))
        for x in range(0, w, s):
            d.line([(x, 0), (x, h)], fill=c1, width=3)
        for y in range(0, h, s):
            d.line([(0, y), (w, y)], fill=c1, width=3)
    else:
        d.rectangle([(0, 0), (w, h)], fill=c0)
        r = int(rng.integers(8, 25))
        for _ in range(30):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            d.ellipse([(x, y), (x + 2 * r, y + 2 * r)], fill=c1)
    return img


def _street(rng, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (w, h)], fill=(90, 90, 90))
    d.rectangle([(0, int(h * 0.6)), (w, h)], fill=(60, 60, 60))
    for i in range(0, w, 24):
        d.line([(i, int(h * 0.6)), (i, h)], fill=(255, 255, 255), width=6)
    for _ in range(3):
        x0 = int(rng.integers(0, int(0.5 * w)))
        y0 = int(rng.integers(int(0.15 * h), int(0.45 * h)))
        d.rectangle([(x0, y0), (x0 + 100, y0 + 60)], fill=(220, 30, 30))
    return img
def _person(rng, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    bg = tuple(int(c) for c in rng.integers(60, 200, 3))
    d.rectangle([(0, 0), (w, h)], fill=bg)
    skin = tuple(int(c) for c in (int(rng.integers(200, 235)),
                                  int(rng.integers(160, 200)),
                                  int(rng.integers(130, 175))))
    shirt = tuple(int(c) for c in rng.integers(40, 160, 3))
    d.ellipse((w // 2 - 60, h // 8, w // 2 + 60, h // 2 + 30), fill=skin)
    d.rectangle((w // 2 - 90, h // 3, w // 2 + 90, h), fill=shirt)
    d.rectangle((w // 2 - 170, h // 3, w // 2 - 90, h - 60), fill=shirt)
    d.rectangle((w // 2 + 90, h // 3, w // 2 + 170, h - 60), fill=shirt)
    return img


def _animal(rng, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    bg = tuple(int(c) for c in rng.integers(60, 190, 3))
    d.rectangle([(0, 0), (w, h)], fill=bg)
    fur = tuple(int(c) for c in rng.integers(90, 210, 3))
    d.ellipse((w // 3, h // 6, 2 * w // 3, h // 2), fill=fur)
    d.ellipse((w // 4, h // 4, w // 3, h // 2), fill=fur)
    d.ellipse((2 * w // 3, h // 4, 3 * w // 4, h // 2), fill=fur)
    d.rectangle((w // 3, h // 2, 2 * w // 3, int(0.85 * h)), fill=fur)
    return img


def _room(rng, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    wall = tuple(int(c) for c in rng.integers(150, 230, 3))
    d.rectangle([(0, 0), (w, h)], fill=wall)
    d.rectangle((w // 6, h // 6, 5 * w // 6, 5 * h // 6),
                fill=tuple(int(c) for c in rng.integers(60, 150, 3)))
    d.rectangle((w // 2 - 24, 5 * h // 6 - 80, w // 2 + 24, 5 * h // 6),
                fill=(60, 60, 80))
    return img


def _document(rng, w, h):
    img = Image.new("RGB", (w, h), color=(250, 248, 240))
    d = ImageDraw.Draw(img)
    for _ in range(30):
        y = int(rng.integers(10, h - 10))
        d.line([(10, y), (w - 10, y)], fill=(90, 90, 90), width=4)
    return img


def _sky(rng, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        c = tuple(int(80 + 120 * (1 - t)) for _ in range(3))
        d.line([(0, y), (w, y)], fill=c)
    d.ellipse((int(0.2 * w), int(0.1 * h), int(0.45 * w), int(0.35 * h)),
              fill=(255, 250, 220))
    return img


_GENERATORS = {
    "flat": _flat, "gradient": _gradient, "noise": _noise, "pattern": _pattern,
    "street": _street, "person": _person, "animal": _animal, "room": _room,
    "document": _document, "sky": _sky,
}


def generate(output_dir: Path, count: int, seed: int = 7, size: int = 320) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(_GENERATORS)
    for i in range(count):
        w = h = size
        kind = names[int(rng.integers(0, len(names)))]
        gen = _GENERATORS[kind]
        if kind == "gradient":
            img = gen(rng, w, h, horizontal=bool(rng.integers(0, 2)))
        else:
            img = gen(rng, w, h)
        img.convert("RGB").save(output_dir / f"{kind}_{i:05d}.jpg", quality=85)
    print(f"Generated {count} synthetic negatives in {output_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/negatives_synthetic")
    ap.add_argument("--count", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()
    generate(Path(args.out), args.count, args.seed, args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())