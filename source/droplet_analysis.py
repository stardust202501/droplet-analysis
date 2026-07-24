"""Cellpose-based droplet segmentation and fluorescence classification."""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
PROJECT_DIR = Path(__file__).resolve().parent
NUMBA_CACHE_DIR = PROJECT_DIR / ".numba_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
NUMBA_CACHE_DIR.mkdir(exist_ok=True)
from sklearn.cluster import KMeans

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DROPLET_COLUMNS = [
    "relative_path",
    "droplet_id",
    "centroid_x",
    "centroid_y",
    "radius_px",
    "area_px",
    "edge_score",
    "edge_touching",
    "raw_mean_gray",
    "raw_median_gray",
    "corrected_mean_gray",
    "local_background_gray",
    "local_contrast_gray",
]


@dataclass(frozen=True)
class AnalysisConfig:
    diameter_px: float = 40.0
    min_area_px: float = 120.0
    max_area_px: float = 0.0
    flow_threshold: float = 1.4
    cellprob_threshold: float = 0.5
    inner_measurement_ratio: float = 0.65
    exclude_edge_droplets: bool = True
    random_state: int = 42


@dataclass
class DropletRegion:
    contour: np.ndarray
    centroid_x: float
    centroid_y: float
    area_px: float
    edge_score: float


@dataclass
class ImageSummary:
    relative_path: str
    total_detected: int
    counted_droplets: int
    positive_droplets: int
    negative_droplets: int
    edge_droplets: int
    negative_center: float
    positive_center: float
    cluster_separation: float
    cluster_quality: str
    focus_score: float
    dynamic_range: float
    saturation_fraction: float
    green_dominance: float
    droplet_csv: str
    overlay_image: str
    status: str = "ok"
    error: str = ""


def load_image(path: Path | str) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path | str, image_rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    extension = suffix if suffix in {".png", ".jpg", ".jpeg"} else ".png"
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(extension, image_bgr)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))
    return path


def normalize_uint8(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = image.astype(np.float32)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) * 255.0 / (hi - lo)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def build_cellpose_inputs(
    image_rgb: np.ndarray,
    diameter_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create complementary green-channel inputs for Cellpose only."""
    green = image_rgb[:, :, 1]
    base = normalize_uint8(green, 1.0, 99.0)
    diameter = max(8.0, float(diameter_px))
    background = cv2.GaussianBlur(base, (0, 0), diameter * 2.2)
    flat = normalize_uint8(
        base.astype(np.float32) - background.astype(np.float32) + np.median(background),
        1.0,
        99.0,
    )
    enhanced = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(flat)
    bright_excess = np.maximum(
        flat.astype(np.float32) - float(np.percentile(flat, 98.8)), 0.0
    )
    halo = cv2.GaussianBlur(bright_excess, (0, 0), diameter * 0.8)
    halo_suppressed = normalize_uint8(flat.astype(np.float32) - 1.6 * halo, 1.0, 98.5)
    halo_suppressed = cv2.createCLAHE(
        clipLimit=1.7, tileGridSize=(8, 8)
    ).apply(halo_suppressed)
    return base, enhanced, halo_suppressed


def _cellpose_model(model_type: str):
    from cellpose import models

    return models.Cellpose(gpu=False, model_type=model_type)


def _run_cellpose(
    model,
    image: np.ndarray,
    config: AnalysisConfig,
    flow_threshold: float | None = None,
    cellprob_threshold: float | None = None,
) -> np.ndarray:
    masks, *_ = model.eval(
        image,
        diameter=float(config.diameter_px),
        channels=[0, 0],
        flow_threshold=(
            float(config.flow_threshold) if flow_threshold is None else float(flow_threshold)
        ),
        cellprob_threshold=(
            float(config.cellprob_threshold)
            if cellprob_threshold is None
            else float(cellprob_threshold)
        ),
    )
    return masks.astype(np.int32)


def _filter_masks(masks: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    expected_area = np.pi * (max(4.0, float(config.diameter_px)) / 2.0) ** 2
    lower = max(float(config.min_area_px), expected_area * 0.10)
    upper = (
        float(config.max_area_px)
        if float(config.max_area_px) > 0
        else expected_area * 2.50
    )
    filtered = np.zeros(masks.shape, dtype=np.int32)
    next_label = 1
    for label in np.unique(masks):
        if label <= 0:
            continue
        region = masks == label
        area = float(np.count_nonzero(region))
        if lower <= area <= upper:
            filtered[region] = next_label
            next_label += 1
    return filtered


def _merge_cellpose_masks(mask_sets: list[np.ndarray], diameter_px: float) -> np.ndarray:
    merged = np.zeros(mask_sets[0].shape, dtype=np.int32)
    centres: list[tuple[float, float]] = []
    next_label = 1
    duplicate_distance = max(4.0, float(diameter_px) * 0.42)
    for masks in mask_sets:
        for label in np.unique(masks):
            if label <= 0:
                continue
            region = masks == label
            ys, xs = np.where(region)
            if not xs.size:
                continue
            centre = (float(np.mean(xs)), float(np.mean(ys)))
            if any(np.hypot(centre[0] - old[0], centre[1] - old[1]) < duplicate_distance for old in centres):
                continue
            if float(np.mean(merged[region] > 0)) > 0.20:
                continue
            available = region & (merged == 0)
            if np.count_nonzero(available) < float(np.pi * 3.0**2):
                continue
            merged[available] = next_label
            centres.append(centre)
            next_label += 1
    return merged


def detect_droplets(
    image_rgb: np.ndarray,
    config: AnalysisConfig,
) -> tuple[list[DropletRegion], np.ndarray]:
    """Segment droplets from enhanced images with Cellpose instance masks."""
    green = image_rgb[:, :, 1]
    green_float = green.astype(np.float32)
    background = cv2.GaussianBlur(
        green_float, (0, 0), max(35.0, float(config.diameter_px) * 1.75)
    )
    _base, enhanced, halo_suppressed = build_cellpose_inputs(
        image_rgb, config.diameter_px
    )
    primary_model = _cellpose_model("cyto3")
    recovery_model = _cellpose_model("nuclei")
    mask_sets = [
        _filter_masks(
            _run_cellpose(
                primary_model,
                image_rgb,
                config,
                flow_threshold=0.4,
                cellprob_threshold=-1.0,
            ),
            config,
        ),
        _filter_masks(_run_cellpose(recovery_model, halo_suppressed, config), config),
        _filter_masks(_run_cellpose(recovery_model, enhanced, config), config),
    ]
    masks = _merge_cellpose_masks(mask_sets, config.diameter_px)
    regions: list[DropletRegion] = []
    for label in np.unique(masks):
        if label <= 0:
            continue
        mask = (masks == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        moments = cv2.moments(contour)
        if area <= 0 or not moments["m00"]:
            continue
        regions.append(
            DropletRegion(
                contour=contour,
                centroid_x=float(moments["m10"] / moments["m00"]),
                centroid_y=float(moments["m01"] / moments["m00"]),
                area_px=area,
                edge_score=1.0,
            )
        )
    regions.sort(key=lambda region: (region.centroid_y, region.centroid_x))
    return regions, background


def image_quality_metrics(image_rgb: np.ndarray) -> dict[str, float]:
    green = image_rgb[:, :, 1]
    channel_means = np.mean(image_rgb.astype(np.float32), axis=(0, 1))
    green_dominance = float(
        channel_means[1] / max(float(channel_means[0]), float(channel_means[2]), 1.0)
    )
    return {
        "focus_score": float(cv2.Laplacian(green, cv2.CV_64F).var()),
        "dynamic_range": float(np.percentile(green, 99) - np.percentile(green, 1)),
        "saturation_fraction": float(np.mean(green >= 254)),
        "green_dominance": green_dominance,
    }


def measure_droplets(
    green: np.ndarray,
    background: np.ndarray,
    regions: list[DropletRegion],
    config: AnalysisConfig,
    relative_path: str,
) -> pd.DataFrame:
    height, width = green.shape
    global_background = float(np.median(background))
    rows = []
    for droplet_id, region in enumerate(regions, start=1):
        contour = region.contour
        x, y = region.centroid_x, region.centroid_y
        radius = float(np.sqrt(region.area_px / np.pi))
        x_values = contour[:, 0, 0]
        y_values = contour[:, 0, 1]
        edge_touching = bool(
            np.min(x_values) <= 0
            or np.min(y_values) <= 0
            or np.max(x_values) >= width - 1
            or np.max(y_values) >= height - 1
        )

        full_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(full_mask, [contour], -1, 255, cv2.FILLED)
        erosion_size = max(
            1, int(round(radius * (1.0 - float(config.inner_measurement_ratio))))
        )
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1)
        )
        inner_mask = cv2.erode(full_mask, erosion_kernel)
        outer_size = max(2, int(round(radius * 0.50)))
        gap_size = max(1, int(round(radius * 0.05)))
        outer_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (outer_size * 2 + 1, outer_size * 2 + 1)
        )
        gap_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (gap_size * 2 + 1, gap_size * 2 + 1)
        )
        outer_mask = cv2.dilate(full_mask, outer_kernel)
        gap_mask = cv2.dilate(full_mask, gap_kernel)
        annulus_mask = (outer_mask > 0) & (gap_mask == 0)
        values = green[inner_mask > 0].astype(np.float32)
        annulus_values = green[annulus_mask].astype(np.float32)
        raw_mean = float(np.mean(values)) if values.size else float("nan")
        raw_median = float(np.median(values)) if values.size else float("nan")
        local_background = (
            float(np.mean(annulus_values)) if annulus_values.size else raw_mean
        )
        local_contrast = raw_mean - local_background
        centre_x = int(np.clip(round(x), 0, width - 1))
        centre_y = int(np.clip(round(y), 0, height - 1))
        corrected_mean = raw_mean - float(background[centre_y, centre_x]) + global_background
        rows.append(
            {
                "relative_path": relative_path,
                "droplet_id": droplet_id,
                "centroid_x": round(x, 3),
                "centroid_y": round(y, 3),
                "radius_px": round(radius, 3),
                "area_px": round(region.area_px, 3),
                "edge_score": round(region.edge_score, 5),
                "edge_touching": edge_touching,
                "raw_mean_gray": round(raw_mean, 5),
                "raw_median_gray": round(raw_median, 5),
                "corrected_mean_gray": round(corrected_mean, 5),
                "local_background_gray": round(local_background, 5),
                "local_contrast_gray": round(local_contrast, 5),
            }
        )
    return pd.DataFrame(rows, columns=DROPLET_COLUMNS)


def classify_kmeans(
    table: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    result = table.copy()
    result["cluster"] = pd.Series(dtype="Int64")
    result["kmeans_feature"] = np.nan
    result["classification"] = "excluded_edge"
    valid = ~result["edge_touching"] if config.exclude_edge_droplets else np.ones(len(result), dtype=bool)
    valid_indices = result.index[valid].to_numpy()
    values = result.loc[valid, "local_contrast_gray"].to_numpy(dtype=float)
    features = np.square(np.maximum(values, 0.0))

    empty_stats: dict[str, float | str] = {
        "negative_center": float("nan"),
        "positive_center": float("nan"),
        "cluster_separation": float("nan"),
        "cluster_quality": "insufficient",
    }
    if values.size == 0:
        return result, empty_stats
    if values.size < 2 or np.unique(np.round(features, 6)).size < 2:
        result.loc[valid_indices, "cluster"] = 0
        result.loc[valid_indices, "kmeans_feature"] = features
        result.loc[valid_indices, "classification"] = "negative"
        centre = float(np.mean(values))
        empty_stats.update(
            {
                "negative_center": centre,
                "positive_center": float("nan"),
                "cluster_separation": 0.0,
                "cluster_quality": "single_intensity",
            }
        )
        return result, empty_stats

    model = KMeans(n_clusters=2, random_state=config.random_state, n_init=20)
    labels = model.fit_predict(features.reshape(-1, 1))
    feature_centers = model.cluster_centers_.reshape(-1)
    negative_cluster = int(np.argmin(feature_centers))
    positive_cluster = int(np.argmax(feature_centers))
    classes = np.where(labels == positive_cluster, "positive", "negative")
    result.loc[valid_indices, "cluster"] = labels
    result.loc[valid_indices, "kmeans_feature"] = features
    result.loc[valid_indices, "classification"] = classes

    cluster_stds = []
    original_centers = []
    for cluster_id in range(2):
        cluster_values = values[labels == cluster_id]
        cluster_stds.append(float(np.std(cluster_values)) if cluster_values.size else 0.0)
        original_centers.append(float(np.mean(cluster_values)) if cluster_values.size else 0.0)
    pooled_std = float(np.sqrt(np.mean(np.square(cluster_stds))))
    gap = float(original_centers[positive_cluster] - original_centers[negative_cluster])
    separation = gap / max(pooled_std, 1e-6)
    quality = "good" if separation >= 2.0 and gap >= 2.0 else "low"
    stats: dict[str, float | str] = {
        "negative_center": float(original_centers[negative_cluster]),
        "positive_center": float(original_centers[positive_cluster]),
        "cluster_separation": separation,
        "cluster_quality": quality,
    }
    return result, stats


def make_overlay(
    image_rgb: np.ndarray,
    table: pd.DataFrame,
    regions: list[DropletRegion],
) -> np.ndarray:
    overlay = image_rgb.copy()
    colors = {
        "positive": (255, 62, 62),
        "negative": (0, 220, 220),
        "excluded_edge": (255, 205, 45),
    }
    for row, region in zip(table.itertuples(index=False), regions):
        color = colors.get(str(row.classification), (255, 205, 45))
        cv2.drawContours(overlay, [region.contour], -1, color, 1, cv2.LINE_AA)
    return overlay


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def analyze_image(
    image_path: Path | str,
    output_root: Path | str,
    input_root: Path | str,
    config: Optional[AnalysisConfig] = None,
) -> ImageSummary:
    config = config or AnalysisConfig()
    image_path = Path(image_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    relative_path = image_path.relative_to(input_root)
    relative_parent = relative_path.parent
    image_output_dir = output_root / relative_parent
    image_output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = load_image(image_path)
    green = image_rgb[:, :, 1]
    quality = image_quality_metrics(image_rgb)
    if quality["green_dominance"] < 2.0:
        stem = _safe_stem(image_path)
        droplet_csv = image_output_dir / f"{stem}_droplets.csv"
        overlay_path = image_output_dir / f"{stem}_overlay.png"
        empty_table = pd.DataFrame(
            columns=DROPLET_COLUMNS + ["cluster", "kmeans_feature", "classification"]
        )
        empty_table.to_csv(droplet_csv, index=False, encoding="utf-8-sig")
        save_rgb(overlay_path, image_rgb)
        return ImageSummary(
            relative_path=relative_path.as_posix(),
            total_detected=0,
            counted_droplets=0,
            positive_droplets=0,
            negative_droplets=0,
            edge_droplets=0,
            negative_center=float("nan"),
            positive_center=float("nan"),
            cluster_separation=float("nan"),
            cluster_quality="not_applicable",
            focus_score=quality["focus_score"],
            dynamic_range=quality["dynamic_range"],
            saturation_fraction=quality["saturation_fraction"],
            green_dominance=quality["green_dominance"],
            droplet_csv=str(droplet_csv),
            overlay_image=str(overlay_path),
            status="skipped_non_green",
        )
    regions, background = detect_droplets(image_rgb, config)
    table = measure_droplets(green, background, regions, config, relative_path.as_posix())
    table, cluster_stats = classify_kmeans(table, config)

    stem = _safe_stem(image_path)
    droplet_csv = image_output_dir / f"{stem}_droplets.csv"
    overlay_path = image_output_dir / f"{stem}_overlay.png"
    table.to_csv(droplet_csv, index=False, encoding="utf-8-sig")
    save_rgb(overlay_path, make_overlay(image_rgb, table, regions))

    counted = table[table["classification"].isin(["positive", "negative"])]
    positive = int(np.sum(counted["classification"] == "positive"))
    negative = int(np.sum(counted["classification"] == "negative"))
    edge = int(np.sum(table["edge_touching"]))
    if positive + negative != len(counted):
        raise RuntimeError("Classification count consistency check failed.")

    return ImageSummary(
        relative_path=relative_path.as_posix(),
        total_detected=int(len(table)),
        counted_droplets=int(len(counted)),
        positive_droplets=positive,
        negative_droplets=negative,
        edge_droplets=edge,
        negative_center=float(cluster_stats["negative_center"]),
        positive_center=float(cluster_stats["positive_center"]),
        cluster_separation=float(cluster_stats["cluster_separation"]),
        cluster_quality=str(cluster_stats["cluster_quality"]),
        focus_score=quality["focus_score"],
        dynamic_range=quality["dynamic_range"],
        saturation_fraction=quality["saturation_fraction"],
        green_dominance=quality["green_dominance"],
        droplet_csv=str(droplet_csv),
        overlay_image=str(overlay_path),
        status=(
            "ok"
            if str(cluster_stats["cluster_quality"]) == "good"
            else "review_cluster"
        ),
    )


def scan_images(input_root: Path | str, output_root: Path | str | None = None) -> list[Path]:
    input_root = Path(input_root).resolve()
    output_path = Path(output_root).resolve() if output_root else None
    images = []
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = path.resolve()
        if output_path is not None and (resolved == output_path or output_path in resolved.parents):
            continue
        images.append(path)
    return sorted(images, key=lambda item: item.relative_to(input_root).as_posix().lower())


def analyze_single_image(
    image_path: Path | str,
    output_root: Path | str,
    config: Optional[AnalysisConfig] = None,
) -> pd.DataFrame:
    image_path = Path(image_path).resolve()
    if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Select a supported image file: {image_path}")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = analyze_image(
        image_path,
        output_root,
        image_path.parent,
        config or AnalysisConfig(),
    )
    table = pd.DataFrame([asdict(summary)])
    table.to_csv(output_root / "batch_summary.csv", index=False, encoding="utf-8-sig")
    return table


def analyze_folder(
    input_root: Path | str,
    output_root: Path | str,
    config: Optional[AnalysisConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    config = config or AnalysisConfig()
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    images = scan_images(input_root, output_root)
    summaries = []
    total = len(images)
    for index, image_path in enumerate(images, start=1):
        if cancel_check and cancel_check():
            break
        if progress_callback:
            progress_callback(index - 1, total, image_path.name)
        try:
            summary = analyze_image(image_path, output_root, input_root, config)
        except Exception as exc:
            relative = image_path.relative_to(input_root).as_posix()
            summary = ImageSummary(
                relative_path=relative,
                total_detected=0,
                counted_droplets=0,
                positive_droplets=0,
                negative_droplets=0,
                edge_droplets=0,
                negative_center=float("nan"),
                positive_center=float("nan"),
                cluster_separation=float("nan"),
                cluster_quality="error",
                focus_score=float("nan"),
                dynamic_range=float("nan"),
                saturation_fraction=float("nan"),
                green_dominance=float("nan"),
                droplet_csv="",
                overlay_image="",
                status="error",
                error=str(exc),
            )
        summaries.append(asdict(summary))
        pd.DataFrame(summaries).to_csv(
            output_root / "batch_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if progress_callback:
            progress_callback(index, total, image_path.name)
    return pd.DataFrame(summaries)
