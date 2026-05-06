"""Reusable modules for coastal oil-spill risk screening."""

from .density_rasters import MeanRasterAggregator, RasterGroup
from .hotspots import HotspotConfig, RunOptions, run_hotspot_extraction
from .gridding import export_oscar_uv_geotiff, standardize_oscar_uv_netcdf
from .models import OscarDownloadConfig, StudyArea, infer_study_area_from_hotspots
from .oscar_workflow import download_oscar_for_periods
from .periods import seasonal_periods
from .podaac import build_podaac_downloader_cmd, run_podaac_downloader, write_earthdata_netrc
from .trajectory import (
    CoastalRiskResult,
    CurrentField,
    current_field_from_geotiff,
    current_field_from_netcdf,
    HotspotSource,
    SimulationConfig,
    estimate_coastal_risk,
    simulate_particles,
)

__all__ = [
    "CoastalRiskResult",
    "CurrentField",
    "HotspotConfig",
    "HotspotSource",
    "MeanRasterAggregator",
    "OscarDownloadConfig",
    "RasterGroup",
    "RunOptions",
    "SimulationConfig",
    "StudyArea",
    "build_podaac_downloader_cmd",
    "download_oscar_for_periods",
    "standardize_oscar_uv_netcdf",
    "export_oscar_uv_geotiff",
    "current_field_from_geotiff",
    "current_field_from_netcdf",
    "estimate_coastal_risk",
    "infer_study_area_from_hotspots",
    "run_podaac_downloader",
    "run_hotspot_extraction",
    "seasonal_periods",
    "simulate_particles",
    "write_earthdata_netrc",
]
