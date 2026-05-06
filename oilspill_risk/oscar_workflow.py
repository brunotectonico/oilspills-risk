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

from .gridding import standardize_oscar_uv_netcdf
from .models import OscarDownloadConfig, StudyArea, validate_study_area
from .podaac import run_podaac_downloader


def raw_oscar_files(output_dir: Path) -> set[Path]:
    """Return raw OSCAR NetCDF files, excluding clipped/derived outputs."""
    return {
        path.resolve()
        for path in output_dir.glob("*.nc")
        if path.name.lower().startswith("oscar") and "clip" not in path.name.lower()
    }


def extract_data_date(raw_file: Path) -> str:
    """Extract a YYYYMMDD data date from an OSCAR filename."""
    match = re.search(r"(?:19|20)\d{6}", raw_file.name)
    if not match:
        raise ValueError(f"Could not parse data date from filename: {raw_file.name}")
    return match.group(0)


def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    periods: list[tuple[date, date, str]],
    *,
    standardize: bool = False,
) -> list[Path]:
    """Download OSCAR files for date periods and optionally clip/standardize each one."""
    outputs: list[Path] = []
    area = validate_study_area(area)

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
        new_files = [path for path in current_raw if path.resolve() not in existing_raw]
        if not new_files:
            raise FileNotFoundError(f"No new raw NetCDF files found in {cfg.output_dir} for period {period_id}")

        for raw_file in new_files:
            if standardize:
                data_date = extract_data_date(raw_file)
                output_nc = cfg.output_dir / f"oscar_uv_clip_{data_date}.nc"
                outputs.append(
                    standardize_oscar_uv_netcdf(
                        raw_file,
                        output_nc,
                        area,
                        u_var=cfg.u_var,
                        v_var=cfg.v_var,
                    )
                )
            else:
                outputs.append(raw_file)

    return outputs
