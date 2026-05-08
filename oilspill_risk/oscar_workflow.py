"""High-level OSCAR download and standardization workflow.

The implementation is split across focused modules:
``models.py`` for shared dataclasses, ``periods.py`` for date windows,
``podaac.py`` for downloader calls, ``gridding.py`` for NetCDF/GeoTIFF grid
handling, and ``oscar_workflow.py`` for high-level orchestration.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from .gridding import export_oscar_uv_geotiff, standardize_oscar_uv_netcdf
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


def _raw_file_date(raw_file: Path) -> date:
    """Return the date encoded in an OSCAR raw filename."""
    return datetime.strptime(extract_data_date(raw_file), "%Y%m%d").date()


def _raw_files_for_period(output_dir: Path, period_start: date, period_end: date) -> list[Path]:
    """Return raw OSCAR files whose encoded date falls inside a period."""
    period_files: list[Path] = []
    for raw_file in raw_oscar_files(output_dir):
        data_date = _raw_file_date(raw_file)
        if period_start <= data_date <= period_end:
            period_files.append(raw_file)
    return sorted(period_files, key=lambda path: (extract_data_date(path), path.name))


# def _period_suffix(period_id: str) -> str:
#     """Return a filename-safe period suffix from seasonal_periods output."""
#     return re.sub(r"[^A-Za-z0-9_-]+", "_", period_id).strip("_")


def _append_unique(paths: list[Path], seen: set[Path], path: Path) -> None:
    """Append a resolved path once while preserving insertion order."""
    resolved = path.resolve()
    if resolved not in seen:
        paths.append(path)
        seen.add(resolved)


def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    periods: list[tuple[date, date, str]],
    area: StudyArea | None = None,
    *,
    standardize: bool = False,
) -> list[Path]:
    """Download OSCAR files for date periods and optionally clip/standardize each one.

    Returned paths are period-scoped and include already-present files that match
    the requested dates. In standardize mode, each derived NetCDF filename embeds
    the ``period_id`` from ``seasonal_periods()`` before its data date, and each
    raw file contributes its standardized NetCDF plus the paired U/V GeoTIFF
    outputs. Duplicate paths are removed while preserving processing order.
    """
    raw_output_dir = cfg.output_dir / "raw_oscar_files"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = cfg.output_dir / "oscar_files_ready"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    outputs: list[Path] = []
    seen_outputs: set[Path] = set()
    name_suffix = "uv_std"
    if area is not None:
        area = validate_study_area(area)
        name_suffix = "uv_std_clip"

    for period_start, period_end, period_id in periods:
        run_podaac_downloader(
            collection=cfg.podaac_collection,
            output_dir=raw_output_dir,
            start_date=period_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_date=period_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            bbox=None if standardize else area,
            dry_run=False,
        )

        period_raw = _raw_files_for_period(raw_output_dir, period_start, period_end)
        if not period_raw:
            raise FileNotFoundError(f"No raw OSCAR files found for period {period_id} in {raw_output_dir}")

        for raw_file in period_raw:
            if not standardize:
                _append_unique(outputs, seen_outputs, raw_file)
                continue

            data_date = extract_data_date(raw_file)
            output_nc = final_dir / f"oscar_{name_suffix}_{period_id}_{data_date}.nc"#{_period_suffix(period_id)}_{data_date}.nc"
            if not output_nc.exists():
                standardize_oscar_uv_netcdf(raw_file, output_nc, area, u_var=cfg.u_var, v_var=cfg.v_var)
            _append_unique(outputs, seen_outputs, output_nc)

            u_tif = final_dir / f"{output_nc.stem}_{cfg.u_var}.tif"
            v_tif = final_dir / f"{output_nc.stem}_{cfg.v_var}.tif"
            if not u_tif.exists() or not v_tif.exists():
                u_tif, v_tif = export_oscar_uv_geotiff(
                    output_nc, final_dir, area, u_var=cfg.u_var, v_var=cfg.v_var
                )
            _append_unique(outputs, seen_outputs, u_tif)
            _append_unique(outputs, seen_outputs, v_tif)
            #Cleaning raw files
            if output_nc.exists():
             try:
                 raw_file.unlink() # deleting file: _final_{data_date}.nc
                 # print(f"Deleted: {raw_file.name}")
             except Exception as e:
                 print(f"{raw_file} could not be deleted: {e}")

    return outputs
