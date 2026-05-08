"""Reusable modules for coastal oil-spill risk screening."""

from .coastlines import CoastPointSet, coast_points_from_shapefile
from .density_rasters import MeanRasterAggregator, RasterGroup
from .exposure import (
    CoastwardExposureResult,
    aggregate_exposure_probabilities,
    aggregate_exposure_probabilities_by_period,
    aggregate_exposure_probabilities_for_periods,
    coastward_exposure_probability,
    exposure_from_geotiff,
    exposure_from_netcdf,
    extract_period_id_from_path,
    nearest_coast_points,
    write_exposure_geotiff,
)
from .hotspots import HotspotConfig, RunOptions, run_hotspot_extraction
from .gridding import export_oscar_uv_geotiff, standardize_oscar_uv_netcdf
from .models import OscarDownloadConfig, StudyArea, infer_study_area_from_hotspots, validate_study_area
from .oscar_workflow import download_oscar_for_periods, extract_data_date, raw_oscar_files
from .periods import analysis_periods, seasonal_periods
from .podaac import build_podaac_downloader_cmd, run_podaac_downloader, write_earthdata_netrc
from .trajectory import (
    CoastalRiskResult,
    CurrentField,
    current_field_from_geotiff,
    current_field_from_netcdf,
    HotspotSource,
    SimulationConfig,
    estimate_coastal_risk,
    estimate_coastal_risk_for_sources,
    hotspot_sources_from_records,
    simulate_particles,
)

__all__ = [
    "CoastPointSet",
    "CoastalRiskResult",
    "CoastwardExposureResult",
    "CurrentField",
    "HotspotConfig",
    "HotspotSource",
    "MeanRasterAggregator",
    "OscarDownloadConfig",
    "RasterGroup",
    "RunOptions",
    "SimulationConfig",
    "StudyArea",
    "aggregate_exposure_probabilities",
    "analysis_periods",
    "aggregate_exposure_probabilities_by_period",
    "aggregate_exposure_probabilities_for_periods",
    "build_podaac_downloader_cmd",
    "coast_points_from_shapefile",
    "coastward_exposure_probability",
    "download_oscar_for_periods",
    "standardize_oscar_uv_netcdf",
    "export_oscar_uv_geotiff",
    "exposure_from_geotiff",
    "exposure_from_netcdf",
    "extract_data_date",
    "extract_period_id_from_path",
    "current_field_from_geotiff",
    "current_field_from_netcdf",
    "estimate_coastal_risk",
    "estimate_coastal_risk_for_sources",
    "hotspot_sources_from_records",
    "infer_study_area_from_hotspots",
    "nearest_coast_points",
    "raw_oscar_files",
    "run_podaac_downloader",
    "run_hotspot_extraction",
    "seasonal_periods",
    "simulate_particles",
    "validate_study_area",
    "write_earthdata_netrc",
    "write_exposure_geotiff",
]





    
    
    


