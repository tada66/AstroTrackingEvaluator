#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class StarMeasure:
    fwhm_px: float
    elongation: float
    peak_value: float


@dataclass
class ImageScore:
    image: str
    stars_used: int
    median_fwhm_px: float
    median_elongation: float
    trail_index_px: float
    p95_trail_length_px: float
    tracking_score_0_100: float


def detect_star_candidates(gray: np.ndarray, max_candidates: int = 1200) -> np.ndarray:
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0, 0), 3.0)
    enhanced = g1 - g2

    median = float(np.median(enhanced))
    mad = float(np.median(np.abs(enhanced - median))) + 1e-6
    sigma_est = 1.4826 * mad
    threshold = median + 5.0 * sigma_est

    local_max = enhanced == cv2.dilate(enhanced, np.ones((3, 3), np.uint8))
    mask = local_max & (enhanced > threshold)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    strengths = enhanced[ys, xs]
    order = np.argsort(strengths)[::-1]
    ys = ys[order]
    xs = xs[order]

    kept: list[tuple[int, int]] = []
    min_dist2 = 5 * 5
    for y, x in zip(ys, xs):
        if any((y - ky) * (y - ky) + (x - kx) * (x - kx) < min_dist2 for ky, kx in kept):
            continue
        kept.append((int(y), int(x)))
        if len(kept) >= max_candidates:
            break
    return np.array(kept, dtype=np.int32)


def measure_star(gray: np.ndarray, y: int, x: int, radius: int = 6) -> StarMeasure | None:
    h, w = gray.shape
    if y - radius < 0 or x - radius < 0 or y + radius >= h or x + radius >= w:
        return None

    patch = gray[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float64)
    background = np.percentile(patch, 25)
    weights = patch - background
    weights[weights < 0] = 0
    s = float(weights.sum())
    if s <= 0:
        return None

    yy, xx = np.indices(weights.shape)
    cy = float((yy * weights).sum() / s)
    cx = float((xx * weights).sum() / s)

    dy = yy - cy
    dx = xx - cx
    cxx = float((weights * dx * dx).sum() / s)
    cyy = float((weights * dy * dy).sum() / s)
    cxy = float((weights * dx * dy).sum() / s)

    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
    eigvals, _ = np.linalg.eigh(cov)
    l1, l2 = float(eigvals[1]), float(eigvals[0])
    if l2 <= 0 or l1 <= 0:
        return None

    sigma_major = np.sqrt(l1)
    sigma_minor = np.sqrt(l2)
    if not (0.4 <= sigma_minor <= 8.0 and 0.4 <= sigma_major <= 16.0):
        return None

    equiv_sigma = np.sqrt((l1 + l2) / 2.0)
    fwhm = 2.355 * equiv_sigma
    elongation = sigma_major / sigma_minor
    peak = float(patch.max() - background)
    return StarMeasure(fwhm_px=fwhm, elongation=elongation, peak_value=peak)


def estimate_trail_lengths(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 12.0)
    enhanced = gray - blur
    enhanced[enhanced < 0] = 0

    nonzero = enhanced[enhanced > 0]
    if nonzero.size < 200:
        return np.array([], dtype=np.float64)

    threshold = float(np.percentile(nonzero, 99.5))
    if threshold <= 0:
        return np.array([], dtype=np.float64)

    binary = (enhanced >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lengths: list[float] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 4.0 or area > 5000.0:
            continue

        rect = cv2.minAreaRect(cnt)
        side_a, side_b = float(rect[1][0]), float(rect[1][1])
        if side_a <= 0 or side_b <= 0:
            continue

        length = max(side_a, side_b)
        if length > 1.0:
            lengths.append(length)

    return np.array(lengths, dtype=np.float64)


def score_image(path: Path) -> ImageScore:
    gray_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray_u8 is None:
        raise RuntimeError(f"Could not read image: {path}")

    gray = gray_u8.astype(np.float32)
    candidates = detect_star_candidates(gray)

    measures: list[StarMeasure] = []
    for y, x in candidates:
        m = measure_star(gray, int(y), int(x))
        if m is None:
            continue
        if m.peak_value < 6.0:
            continue
        measures.append(m)

    if len(measures) < 15:
        return ImageScore(
            image=path.name,
            stars_used=len(measures),
            median_fwhm_px=float("nan"),
            median_elongation=float("nan"),
            trail_index_px=float("inf"),
            p95_trail_length_px=float("inf"),
            tracking_score_0_100=0.0,
        )

    fwhms = np.array([m.fwhm_px for m in measures], dtype=np.float64)
    elons = np.array([m.elongation for m in measures], dtype=np.float64)
    med_fwhm = float(np.median(fwhms))
    med_elong = float(np.median(elons))
    trail_lengths = estimate_trail_lengths(gray)
    p95_trail_len = float(np.percentile(trail_lengths, 95)) if trail_lengths.size else float("nan")

    # Elongation captures directional tracking drift better than plain FWHM.
    trail_index = med_fwhm * (1.0 - 1.0 / med_elong)
    blur_penalty = max(0.0, med_fwhm - 3.0) * 0.15
    long_trail_penalty = 0.0
    if np.isfinite(p95_trail_len):
        long_trail_penalty = (max(0.0, p95_trail_len - 8.0) / 18.0) ** 2
    composite = (trail_index / 1.45) ** 2 + blur_penalty + long_trail_penalty
    tracking_score = 100.0 / (1.0 + composite)
    tracking_score = float(np.clip(tracking_score, 0.0, 100.0))

    return ImageScore(
        image=path.name,
        stars_used=int(len(measures)),
        median_fwhm_px=med_fwhm,
        median_elongation=med_elong,
        trail_index_px=trail_index,
        p95_trail_length_px=p95_trail_len,
        tracking_score_0_100=tracking_score,
    )


def iter_images(targets: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        p = Path(target)
        if p.is_dir():
            paths.extend(sorted([x for x in p.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]))
        elif p.exists():
            paths.append(p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate astro tracking quality using FWHM-based metrics.")
    parser.add_argument("paths", nargs="+", help="Image files or directories.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = parser.parse_args()

    images = iter_images(args.paths)
    if not images:
        raise SystemExit("No images found.")

    results = [score_image(p) for p in images]
    results.sort(key=lambda r: r.tracking_score_0_100, reverse=True)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    def format_image_name(name: str, width: int = 30) -> str:
        if len(name) <= width:
            return f"{name:<{width}}"
        if width <= 3:
            return name[:width]
        return f"{name[: width - 3]}..."

    print("Tracking quality (higher score is better):")
    for r in results:
        print(
            f"{format_image_name(r.image)} score={r.tracking_score_0_100:6.2f}  "
            f"FWHM={r.median_fwhm_px:5.2f}px  elong={r.median_elongation:4.2f}  "
            f"trail_idx={r.trail_index_px:4.2f}px  p95_trail={r.p95_trail_length_px:5.1f}px  stars={r.stars_used:4d}"
        )


if __name__ == "__main__":
    main()
