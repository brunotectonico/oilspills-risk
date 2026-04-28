"""Extract and cluster tanker-density hotspots from GMTDS ZIP archives.

This script scans ZIP files that contain monthly GeoTIFF rasters, identifies
high-density pixels in a geographic region, clusters those pixels into
hotspots with DBSCAN, and writes both detailed and summary CSV outputs.
"""

from __future__ import annotations

import argparse
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from sklearn.cluster import DBSCAN

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HotspotConfig:
    """Runtime configuration for hotspot extraction."""

    lon_min: float = 40.0
    lon_max: float = 45.0
    lat_min: float = 10.0
    lat_max: float = 14.0
    percentile_threshold: float = 90.0
    min_high_pixels: int = 10
    dbscan_eps: float = 0.1
    dbscan_min_samples: int = 5


@dataclass(frozen=True)
class RunOptions:
    """Input/output options."""

    data_dir: Path
    zip_pattern: str = "*Tankers.zip"
    output_csv: str = "gmtds_tanker_hotspots_multi.csv"
    output_summary_csv: str = "gmtds_tanker_hotspots_monthly_summary.csv"
    limit: int | None = None
    start: int = 0


YEAR_PATTERN = re.compile(r"_(\d{4})")
MONTH_PATTERN = re.compile(r"_(\d{2})_")


def parse_year(zip_name: str) -> str:
    """Extract year from ZIP filename."""
    match = YEAR_PATTERN.search(zip_name)
    if not match:
        raise ValueError(f"Could not parse year from ZIP filename: {zip_name}")
    return match.group(1)


def parse_month(tif_name: str) -> str:
    """Extract month from TIFF filename."""
    match = MONTH_PATTERN.search(tif_name)
    if not match:
        raise ValueError(f"Could not parse month from TIFF filename: {tif_name}")
    return match.group(1)


def pixel_centers(transform: rasterio.Affine, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create longitude/latitude arrays for raster cell centers."""
    height, width = shape
    x_coords = transform.c + (np.arange(width) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(height) + 0.5) * transform.e
    return np.meshgrid(x_coords, y_coords)


def read_density_from_zip(zip_file: zipfile.ZipFile, tif_name: str) -> tuple[np.ndarray, rasterio.Affine]:
    """Read a single-band density raster from a TIFF inside a ZIP into memory."""
    with zip_file.open(tif_name, "r") as tif_stream:
        tif_bytes = tif_stream.read()

    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            density = src.read(1, masked=True).filled(np.nan).astype(float)
            return density, src.transform


def extract_hotspots_from_raster(
    density: np.ndarray,
    transform: rasterio.Affine,
    year: str,
    month: str,
    source_file: str,
    cfg: HotspotConfig,
) -> list[dict[str, float | int | str]]:
    """Detect and cluster high-density pixels from a single monthly raster."""
    valid = density[~np.isnan(density)]
    if valid.size == 0:
        return []

    threshold = np.percentile(valid, cfg.percentile_threshold)
    high_mask = density > threshold

    lon, lat = pixel_centers(transform, density.shape)
    region_mask = (
        (lon >= cfg.lon_min)
        & (lon <= cfg.lon_max)
        & (lat >= cfg.lat_min)
        & (lat <= cfg.lat_max)
    )

    candidate_mask = high_mask & region_mask & ~np.isnan(density)
    rows, cols = np.where(candidate_mask)
    if rows.size < cfg.min_high_pixels:
        return []

    coords = np.column_stack((lon[rows, cols], lat[rows, cols]))
    densities = density[rows, cols]
    labels = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit(coords).labels_

    hotspots: list[dict[str, float | int | str]] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_lon = coords[cluster_mask, 0]
        cluster_lat = coords[cluster_mask, 1]
        cluster_dens = densities[cluster_mask]

        imax = np.argmax(cluster_dens) # maximum density in the cluster

        hotspots.append(
            {
                "year": year,
                "month": month,
                "cluster_id": int(label),
                "lon": float(cluster_lon[imax]),
                "lat": float(cluster_lat[imax]),
                "mean_density": float(np.mean(cluster_dens)),
                "max_density": float(cluster_dens[imax]),
                "n_pixels": int(cluster_lon.size),
                "source_file": source_file,
            }
        )
    
    return hotspots


def iter_tifs(zip_file: zipfile.ZipFile) -> Iterable[str]:
    """Yield TIFF members in a ZIP archive in sorted order."""
    members = [name for name in zip_file.namelist() if name.lower().endswith((".tif", ".tiff"))]
    yield from sorted(members)


def process_zip(zip_path: Path, cfg: HotspotConfig) -> list[dict[str, float | int | str]]:
    """Process all TIFFs in a single ZIP archive."""
    year = parse_year(zip_path.stem)
    LOGGER.info("Processing %s (%s)", zip_path.name, year)

    hotspots: list[dict[str, float | int | str]] = []
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        tif_names = list(iter_tifs(zip_file))
        LOGGER.info("  Found %d monthly files", len(tif_names))

        for tif_name in tif_names:
            month = parse_month(Path(tif_name).name)
            try:
                density, transform = read_density_from_zip(zip_file, tif_name)
                hotspots.extend(
                    extract_hotspots_from_raster(
                        density=density,
                        transform=transform,
                        year=year,
                        month=month,
                        source_file=tif_name,
                        cfg=cfg,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - continue processing on per-file errors
                LOGGER.exception("Error processing %s in %s: %s", tif_name, zip_path.name, exc)

    return hotspots


def save_outputs(hotspots: list[dict[str, float | int | str]], options: RunOptions) -> None:
    """Write detailed and monthly summary CSVs."""
    output_path = options.data_dir / options.output_csv
    summary_path = options.data_dir / options.output_summary_csv

    df_hotspots = pd.DataFrame(hotspots)
    if df_hotspots.empty:
        LOGGER.warning("No hotspots found. No output files were written.")
        return

    df_hotspots.sort_values(["year", "month", "cluster_id"], inplace=True)
    df_hotspots.to_csv(output_path, index=False)
    LOGGER.info(
        "Saved %d hotspots across %d years to %s",
        len(df_hotspots),
        df_hotspots["year"].nunique(),
        output_path,
    )

    summary = (
        df_hotspots.groupby(["year", "month"], as_index=False)
        .agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            n_pixels=("n_pixels", "sum"),
            mean_density=("mean_density", "mean"),
            n_clusters=("cluster_id", "count"),
        )
        .sort_values(["year", "month"])
    )
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved monthly summary to %s", summary_path)


def run(options: RunOptions, cfg: HotspotConfig) -> None:
    """Main processing function."""
    zip_files = sorted(options.data_dir.glob(options.zip_pattern))
    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP files matched pattern '{options.zip_pattern}' in {options.data_dir}"
        )

    end = None if options.limit is None else options.start + options.limit
    selected = zip_files[options.start:end]
    LOGGER.info("Selected %d ZIP files (from %d total)", len(selected), len(zip_files))

    all_hotspots: list[dict[str, float | int | str]] = []
    for zip_path in selected:
        all_hotspots.extend(process_zip(zip_path, cfg))

    save_outputs(all_hotspots, options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GMTDS tanker hotspots")
    parser.add_argument("data_dir", type=Path, help="Directory containing GMTDS ZIP files")
    parser.add_argument("--pattern", default="*Tankers.zip", help="Glob pattern for ZIP files")
    parser.add_argument("--start", type=int, default=0, help="Start index in the sorted ZIP list")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of ZIP files to process")
    parser.add_argument("--output", default="gmtds_tanker_hotspots_multi.csv", help="Detailed output CSV name")
    parser.add_argument(
        "--summary-output",
        default="gmtds_tanker_hotspots_monthly_summary.csv",
        help="Monthly summary output CSV name",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    options = RunOptions(
        data_dir=args.data_dir,
        zip_pattern=args.pattern,
        output_csv=args.output,
        output_summary_csv=args.summary_output,
        start=args.start,
        limit=args.limit,
    )
    cfg = HotspotConfig()

    run(options, cfg)


if __name__ == "__main__":
    main()