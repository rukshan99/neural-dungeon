"""Toy datasets shared by several floors. numpy only, deterministic, tiny.

- make_spirals: the classic interleaved spirals, k classes, 2-D. Not linearly
  separable; an MLP with one hidden layer solves it, a linear model cannot.
- make_moons: two interleaving half circles, binary.
- make_runes: 8x8 glyph bitmaps in 10 classes with jitter and pixel noise. A
  miniature MNIST that needs no download. Class names are dungeon runes.
- train_val_split: shuffle and split.

Every function takes a ``seed`` and returns float32 features and int64 labels.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- 2-D


def make_spirals(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """Interleaved spirals. Returns X (N, 2) float32 and y (N,) int64."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    for c in range(n_classes):
        ix = range(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + rng.standard_normal(n_per_class) * noise
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = c
    return X, y


def make_moons(n: int = 200, noise: float = 0.1, seed: int = 0):
    """Two interleaving half circles. Returns X (n, 2) float32 and y (n,) int64 in {0, 1}."""
    rng = np.random.default_rng(seed)
    n_out = n // 2
    n_in = n - n_out
    outer_t = np.linspace(0, np.pi, n_out)
    inner_t = np.linspace(0, np.pi, n_in)
    outer = np.c_[np.cos(outer_t), np.sin(outer_t)]
    inner = np.c_[1 - np.cos(inner_t), 1 - np.sin(inner_t) - 0.5]
    X = np.vstack([outer, inner]).astype(np.float32)
    X += rng.standard_normal(X.shape).astype(np.float32) * noise
    y = np.r_[np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)]
    return X, y


# ------------------------------------------------------------------------- runes

RUNE_NAMES = ["gate", "stair", "torch", "key", "eye", "skull", "star", "wave", "tower", "loop"]

_RUNE_ART = {
    "gate": [
        "########",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
    ],
    "stair": [
        "#.......",
        "##......",
        "###.....",
        "####....",
        "#####...",
        "######..",
        "#######.",
        "########",
    ],
    "torch": [
        "...##...",
        "..####..",
        "..####..",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
    ],
    "key": [
        "..###...",
        ".#...#..",
        ".#...#..",
        "..###...",
        "...#....",
        "...#.#..",
        "...###..",
        "...#....",
    ],
    "eye": [
        "........",
        "..####..",
        ".#....#.",
        "#..##..#",
        "#..##..#",
        ".#....#.",
        "..####..",
        "........",
    ],
    "skull": [
        "..####..",
        ".######.",
        "##.##.##",
        "##.##.##",
        ".######.",
        "..#..#..",
        "..#..#..",
        "..####..",
    ],
    "star": [
        "...#....",
        "...#....",
        "#..#..#.",
        ".#####..",
        "..###...",
        ".#####..",
        "#.....#.",
        "........",
    ],
    "wave": [
        "........",
        ".##...##",
        "#..#.#..",
        "....#...",
        "........",
        ".##...##",
        "#..#.#..",
        "....#...",
    ],
    "tower": [
        "..#..#..",
        "..####..",
        "..#..#..",
        "..#..#..",
        "..####..",
        "..#..#..",
        "..#..#..",
        ".######.",
    ],
    "loop": [
        "..####..",
        ".#....#.",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        ".#....#.",
        "..####..",
    ],
}


def rune_templates() -> np.ndarray:
    """(10, 8, 8) float32 clean glyphs in the order of RUNE_NAMES."""
    out = np.zeros((len(RUNE_NAMES), 8, 8), dtype=np.float32)
    for i, name in enumerate(RUNE_NAMES):
        rows = _RUNE_ART[name]
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                out[i, r, c] = 1.0 if ch == "#" else 0.0
    return out


def make_runes(n_per_class: int = 100, noise: float = 0.15, jitter: int = 1, seed: int = 0):
    """Noisy, jittered 8x8 rune glyphs. Returns X (N, 8, 8) float32 in [0, 1] and y (N,) int64.

    Each sample is the class template shifted by up to ``jitter`` pixels in each
    direction (wrapping), with Gaussian pixel noise of std ``noise`` added and the
    result clipped to [0, 1]. Samples are shuffled.
    """
    rng = np.random.default_rng(seed)
    templates = rune_templates()
    n_classes = templates.shape[0]
    X = np.zeros((n_per_class * n_classes, 8, 8), dtype=np.float32)
    y = np.zeros(n_per_class * n_classes, dtype=np.int64)
    i = 0
    for c in range(n_classes):
        for _ in range(n_per_class):
            img = templates[c]
            if jitter:
                dr, dc = rng.integers(-jitter, jitter + 1, size=2)
                img = np.roll(np.roll(img, dr, axis=0), dc, axis=1)
            img = img + rng.standard_normal((8, 8)).astype(np.float32) * noise
            X[i] = np.clip(img, 0.0, 1.0)
            y[i] = c
            i += 1
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ------------------------------------------------------------------------ split


def train_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2, seed: int = 0):
    """Shuffle, then split. Returns (X_train, y_train, X_val, y_val)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = int(round(len(X) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
