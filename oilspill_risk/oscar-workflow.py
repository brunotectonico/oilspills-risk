"""OSCAR download and NetCDF utilities using PO.DAAC downloader workflow."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import warnings

import rasterio
from rasterio.transform import from_bounds

from dataclasses import dataclass
from datetime import date, datetime
from calendar import monthranges
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from podaac_io import run_podaac_downloader
from oscar_grid import standardize_oscar_uv_netcdf
from time_windows import seasonal_periods


@dataclass(frozen=True)
class StudyArea:
    """Bounding box used to subset OSCAR fields."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class OscarDownloadConfig:
    """PO.DAAC downloader options for OSCAR data pulls."""

    output_dir: Path
    podaac_collection: str
    u_var: str = "u"
    v_var: str = "v"


def validate_study_area(area: StudyArea) -> StudyArea:
    """Validate and normalize StudyArea bounds."""
    lon_min, lon_max = sorted([float(area.lon_min), float(area.lon_max)])
    lat_min, lat_max = sorted([float(area.lat_min), float(area.lat_max)])

    if not (-180.0 <= lon_min <= 360.0 and -180.0 <= lon_max <= 360.0):
        raise ValueError(f"Invalid longitude bounds: {lon_min}, {lon_max}")
    if not (-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0):
        raise ValueError(f"Invalid latitude bounds: {lat_min}, {lat_max}")
    if lon_min == lon_max or lat_min == lat_max:
        raise ValueError("StudyArea bounds collapse to zero width/height")

    return StudyArea(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

    
def infer_study_area_from_hotspots(
    hotspot_csv: Path,
    lon_col: str = "lon",
    lat_col: str = "lat",
    pad_deg: float = 0.5,
) -> StudyArea:
    """Infer a current-download bounding box from hotspot coordinates."""
    df = pd.read_csv(hotspot_csv)
    area = StudyArea(
        lon_min=float(df[lon_col].min() - pad_deg),
        lon_max=float(df[lon_col].max() + pad_deg),
        lat_min=float(df[lat_col].min() - pad_deg),
        lat_max=float(df[lat_col].max() + pad_deg),
    )
    area = validate_study_area(area)
    print(f"[DEBUG] inferred StudyArea: {area}")
    return area


def _raw_oscar_files(output_dir: Path) -> set[Path]:
    """Track raw OSCAR .nc files only (exclude clipped outputs)."""
    return {
        p.resolve()
        for p in output_dir.glob("*.nc")
        if "clip" not in p.name.lower() and p.name.lower().startswith("oscar")
    }


def _extract_data_date(raw_file: Path) -> str:
    """Extract YYYYMMDD from a raw OSCAR filename."""
    m = re.search(r"(19|20)\d{6}", raw_file.name)
    if not m:
        raise ValueError(f"Could not parse data date from filename: {raw_file.name}")
    return m.group(0)

    
def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    periods: list[tuple[date, date, str]],
    *,
    standardize: bool = False,
) -> list[Path]:
    outputs: list[Path] = []
    area = validate_study_area(area)
    print(f"[DEBUG] download StudyArea: {area}")

    for pstart, pend, pid in periods:
        start_str = pstart.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = pend.strftime('%Y-%m-%dT%H:%M:%SZ')

        downloader_bbox = None if standardize else area
        print(f"[DEBUG] period={pid} downloading...")
        run_podaac_downloader(
            collection=cfg.podaac_collection,
            output_dir=cfg.output_dir,
            start_date=start_str,
            end_date=end_str,
            bbox=downloader_bbox,
            dry_run=False,
        )

        raw_files = sorted(_raw_oscar_files(cfg.output_dir), key=lambda p: p.stat().st_mtime)
        if not raw_files:
            raise FileNotFoundError(f"No raw NetCDF files found in {cfg.output_dir} after period {pid} download")

        raw_out = raw_files[-1]  # latest
        if standardize:
            data_date = _extract_data_date(raw_out)
            std_out = cfg.output_dir / f"oscar_uv_clip_{data_date}.nc"
            if std_out.exists():
                print(f"[DEBUG] clip exists, skipping: {std_out.name}")
            else:
                print(f"[DEBUG] standardize raw={raw_out.name} -> clip={std_out.name}")
                standardize_oscar_uv_netcdf(raw_out, std_out, area, u_var=cfg.u_var, v_var=cfg.v_var)
            outputs.append(std_out)
        else:
            outputs.append(raw_out)

    return outputs