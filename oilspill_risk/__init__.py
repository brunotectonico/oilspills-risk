"""Reusable modules for coastal oil-spill risk screening."""

from .density_rasters import MeanRasterAggregator, RasterGroup
from .hotspots import HotspotConfig, RunOptions, run_hotspot_extraction
from .oscar import (
    OscarDownloadConfig,
    StudyArea,
    download_oscar_for_periods,
    download_oscar_subset,
    infer_study_area_from_hotspots,
    seasonal_periods,
)
from .trajectory import (
    CoastalRiskResult,
    CurrentField,
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
    "download_oscar_for_periods",
    "download_oscar_subset",
    "estimate_coastal_risk",
    "infer_study_area_from_hotspots",
    "run_hotspot_extraction",
    "seasonal_periods",
    "simulate_particles",
]
