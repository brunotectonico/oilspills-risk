"""Backward-compatible facade for OSCAR utilities.

The implementation is split across focused modules:
``models.py`` for shared dataclasses, ``periods.py`` for date windows,
``podaac.py`` for downloader calls, ``gridding.py`` for NetCDF/GeoTIFF grid
handling, and ``oscar_workflow.py`` for high-level orchestration.
"""

from __future__ import annotations

from .gridding import export_oscar_uv_geotiff, standardize_oscar_uv_netcdf
from .models import OscarDownloadConfig, StudyArea, infer_study_area_from_hotspots, validate_study_area
from .oscar_workflow import download_oscar_for_periods, extract_data_date, raw_oscar_files
from .periods import seasonal_periods
from .podaac import build_podaac_downloader_cmd, run_podaac_downloader, write_earthdata_netrc

__all__ = [
    "OscarDownloadConfig",
    "StudyArea",
    "build_podaac_downloader_cmd",
    "download_oscar_for_periods",
    "export_oscar_uv_geotiff",
    "extract_data_date",
    "infer_study_area_from_hotspots",
    "raw_oscar_files",
    "run_podaac_downloader",
    "seasonal_periods",
    "standardize_oscar_uv_netcdf",
    "validate_study_area",
    "write_earthdata_netrc",
]
