"""Hotspot extraction utilities based on GMTDS tanker density rasters."""

from __future__ import annotations

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

from .density_rasters import MeanRasterAggregator, RasterGroup

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
    """Input/output options for hotspot extraction."""

    data_dir: Path
    zip_pattern: str = "*Tankers.zip"
    output_csv: str = "gmtds_tanker_hotspots_multi.csv"
    output_summary_csv: str = "gmtds_tanker_hotspots_monthly_summary.csv"
    mean_raster_dir: str | None = None
    mean_raster_frequency: str = "monthly"
    limit: int | None = None
    start: int = 0
    season_start_month: int | None = None
    season_length_months: int = 3


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


def period_id(year: str, month: str, options: RunOptions) -> str:
    """Build a period identifier (year-season) when seasonal filtering is enabled."""
    if options.season_start_month is None:
        return f"{year}-{month}"

    start = options.season_start_month
    end = ((start + options.season_length_months - 2) % 12) + 1
    return f"{year}-M{start:02d}_M{end:02d}"


def month_in_selected_window(month: int, options: RunOptions) -> bool:
    """Check whether month belongs to optional seasonal window."""
    if options.season_start_month is None:
        return True

    if not 1 <= options.season_start_month <= 12:
        raise ValueError("season_start_month must be between 1 and 12")

    span = max(1, min(12, options.season_length_months))
    allowed = {((options.season_start_month - 1 + i) % 12) + 1 for i in range(span)}
    return month in allowed


def density_group(year: str, month: str, options: RunOptions) -> RasterGroup:
    """Return grouping key for mean density raster output."""
    if options.mean_raster_frequency == "seasonal":
        pid = period_id(year, month, options)
        return RasterGroup(key=pid, filename=f"mean_density_{pid}.tif")
    month_key = f"M{int(month):02d}"
    return RasterGroup(key=month_key, filename=f"mean_density_{month_key}.tif")


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
    options: RunOptions,
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

        imax = np.argmax(cluster_dens)
        hotspots.append(
            {
                "year": year,
                "month": month,
                "period_id": period_id(year, month, options),
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


def process_zip(
    zip_path: Path,
    cfg: HotspotConfig,
    options: RunOptions,
    raster_agg: MeanRasterAggregator | None,
) -> list[dict[str, float | int | str]]:
    """Process all TIFFs in a single ZIP archive."""
    year = parse_year(zip_path.stem)
    LOGGER.info("Processing %s (%s)", zip_path.name, year)

    hotspots: list[dict[str, float | int | str]] = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        tif_names = list(iter_tifs(zip_file))
        LOGGER.info("  Found %d monthly files", len(tif_names))

        for tif_name in tif_names:
            month = parse_month(Path(tif_name).name)
            month_i = int(month)
            if not month_in_selected_window(month_i, options):
                continue

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
                        options=options,
                    )
                )

                if raster_agg is not None:
                    raster_agg.add(density_group(year, month, options), density=density, transform=transform)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Error processing %s in %s: %s", tif_name, zip_path.name, exc)

    return hotspots


def save_outputs(hotspots: list[dict[str, float | int | str]], options: RunOptions) -> None:
    """Write detailed and summary CSV outputs."""
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
        df_hotspots.groupby(["period_id", "year", "month"], as_index=False)
        .agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            n_pixels=("n_pixels", "sum"),
            mean_density=("mean_density", "mean"),
            n_clusters=("cluster_id", "count"),
        )
        .sort_values(["period_id", "year", "month"])
    )
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved monthly summary to %s", summary_path)


def run_hotspot_extraction(options: RunOptions, cfg: HotspotConfig) -> None:
    """Main processing entrypoint for hotspot extraction."""
    zip_files = sorted(options.data_dir.glob(options.zip_pattern))
    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP files matched pattern '{options.zip_pattern}' in {options.data_dir}"
        )

    end = None if options.limit is None else options.start + options.limit
    selected = zip_files[options.start:end]
    LOGGER.info("Selected %d ZIP files (from %d total)", len(selected), len(zip_files))

    raster_agg = MeanRasterAggregator() if options.mean_raster_dir else None

    all_hotspots: list[dict[str, float | int | str]] = []
    for zip_path in selected:
        all_hotspots.extend(process_zip(zip_path, cfg, options, raster_agg=raster_agg))

    save_outputs(all_hotspots, options)

    if raster_agg is not None:
        out_dir = options.data_dir / options.mean_raster_dir
        written = raster_agg.write_all(out_dir)
        LOGGER.info("Saved %d mean density raster(s) in %s", len(written), out_dir)
