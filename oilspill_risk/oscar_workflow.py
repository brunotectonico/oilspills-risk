"""High-level OSCAR download and standardization workflow.

The implementation is split across focused modules:
``models.py`` for shared dataclasses, ``periods.py`` for date windows,
``podaac.py`` for downloader calls, ``gridding.py`` for NetCDF/GeoTIFF grid
handling, and ``oscar_workflow.py`` for high-level orchestration.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .gridding import standardize_oscar_uv_netcdf, export_oscar_uv_geotiff
from .models import OscarDownloadConfig, StudyArea, validate_study_area
from .podaac import run_podaac_downloader


def raw_oscar_files(output_dir: Path) -> set[Path]:
    """Return raw OSCAR NetCDF files, excluding clipped/derived outputs."""
    return {
        path.resolve()
        for path in output_dir.glob("*.nc")
        if path.name.lower().startswith("oscar") and "final" in path.name.lower()
    }


def extract_data_date(raw_file: Path) -> str:
    """Extract a YYYYMMDD data date from an OSCAR filename."""
    match = re.search(r"(?:19|20)\d{6}", raw_file.name)
    if not match:
        raise ValueError(f"Could not parse data date from filename: {raw_file.name}")
    return match.group(0)


def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    periods: list[tuple[date, date, str]],
    area: StudyArea | None = None,
    *,
    standardize: bool = False,
) -> list[Path]:
    """Download OSCAR files for date periods and optionally clip/standardize each one."""
    outputs: list[Path] = []
    new_name = 'uv_std'
    if area is not None:
            area = validate_study_area(area)
            new_name = 'uv_std_clip'

    for period_start, period_end, period_id in periods:
        existing_raw = raw_oscar_files(cfg.output_dir)
        run_podaac_downloader(
            collection=cfg.podaac_collection,
            output_dir=cfg.output_dir,
            start_date=period_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_date=period_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            bbox=None if standardize else area,
            dry_run=False,
        )

        current_raw = sorted(raw_oscar_files(cfg.output_dir), key=lambda path: path.stat().st_mtime)
        for raw_file in current_raw:
            if standardize:
                data_date = extract_data_date(raw_file)
                output_nc = cfg.output_dir / f"oscar_{new_name}_{data_date}.nc"
                
                if output_nc.exists():
                    continue  
                    
                outputs.append(
                    standardize_oscar_uv_netcdf(raw_file, output_nc, area, u_var=cfg.u_var, v_var=cfg.v_var)
                )
                outputs.append(
                    export_oscar_uv_geotiff(output_nc, cfg.output_dir, area)
                )
            else:
                if raw_file.resolve() not in existing_raw:
                    outputs.append(raw_file)
    
        if not outputs:
            raise FileNotFoundError(f"No files to convert in {cfg.output_dir}")

    return outputs
