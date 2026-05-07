"""Shared OSCAR workflow data models and study-area helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    """Validate and normalize a study-area bounding box."""
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
    import pandas as pd

    df = pd.read_csv(hotspot_csv)
    area = StudyArea(
        lon_min=float(df[lon_col].min() - pad_deg),
        lon_max=float(df[lon_col].max() + pad_deg),
        lat_min=float(df[lat_col].min() - pad_deg),
        lat_max=float(df[lat_col].max() + pad_deg),
    )
    return validate_study_area(area)
