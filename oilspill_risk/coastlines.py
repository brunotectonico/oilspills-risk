"""Coastline ingestion and point-sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoastPointSet:
    """Sampled coastline points and optional segment identifiers."""

    points: np.ndarray
    segment_ids: np.ndarray | None = None
    crs: Any | None = None


def _iter_line_geometries(geometry: Any):
    """Yield line-like geometries from points, lines, polygons, or collections."""
    if geometry is None or getattr(geometry, "is_empty", False):
        return

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in {"LineString", "LinearRing"}:
        yield geometry
    elif geom_type == "Polygon":
        yield geometry.exterior
        yield from geometry.interiors
    elif geom_type.startswith("Multi") or geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from _iter_line_geometries(part)


def _points_from_point_geometry(geometry: Any) -> list[tuple[float, float]]:
    """Return coordinates from point-like geometry."""
    if geometry is None or getattr(geometry, "is_empty", False):
        return []

    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if geom_type == "MultiPoint":
        return [(float(point.x), float(point.y)) for point in geometry.geoms]
    return []


def _sample_line(line: Any, spacing: float, include_endpoints: bool) -> list[tuple[float, float]]:
    """Sample a Shapely line at a fixed spacing in the line CRS units."""
    length = float(line.length)
    if length == 0.0:
        coords = list(line.coords)
        return [(float(coords[0][0]), float(coords[0][1]))] if coords else []

    spacing = max(float(spacing), 1e-12)
    distances = list(np.arange(0.0, length, spacing))
    if include_endpoints and (not distances or not np.isclose(distances[-1], length)):
        distances.append(length)

    sampled: list[tuple[float, float]] = []
    for distance in distances:
        point = line.interpolate(float(distance))
        sampled.append((float(point.x), float(point.y)))
    return sampled


def coast_points_from_shapefile(
    path: Path | str,
    *,
    spacing: float = 0.01,
    target_crs: Any | None = None,
    segment_id_col: str | None = None,
    selected_segments: set[Any] | list[Any] | tuple[Any, ...] | None = None,
    include_endpoints: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
) -> CoastPointSet:
    """Read coastline geometries and sample them into points.

    Parameters
    ----------
    path:
        Vector file readable by GeoPandas, such as a Shapefile or GeoPackage.
    spacing:
        Sampling distance in the units of ``target_crs`` when provided, or in
        the source CRS units otherwise. For EPSG:4326 inputs without reprojection,
        this is decimal degrees.
    target_crs:
        Optional CRS passed to ``GeoDataFrame.to_crs`` before sampling. The
        returned points use this CRS, so it must match the current-grid
        coordinates used in exposure calculations.
    segment_id_col:
        Optional attribute column identifying coastline segments.
    selected_segments:
        Optional set/list/tuple of segment IDs to retain before sampling.
    include_endpoints:
        Whether each sampled line should include its end point.
    bbox:
        Optional bounding box ``(minx, miny, maxx, maxy)`` used by GeoPandas
        while reading the vector file.
    """
    import geopandas as gpd

    gdf = gpd.read_file(path, bbox=bbox)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    if selected_segments is not None:
        if segment_id_col is None:
            raise ValueError("segment_id_col is required when selected_segments is provided")
        gdf = gdf[gdf[segment_id_col].isin(set(selected_segments))]

    points: list[tuple[float, float]] = []
    segment_ids: list[Any] = []

    for index, row in gdf.iterrows():
        geometry = row.geometry
        segment_id = row[segment_id_col] if segment_id_col is not None else index

        point_coords = _points_from_point_geometry(geometry)
        for coord in point_coords:
            points.append(coord)
            segment_ids.append(segment_id)

        for line in _iter_line_geometries(geometry):
            for coord in _sample_line(line, spacing=spacing, include_endpoints=include_endpoints):
                points.append(coord)
                segment_ids.append(segment_id)

    if not points:
        raise ValueError(f"No coastline points could be sampled from {path}")

    return CoastPointSet(
        points=np.asarray(points, dtype=float),
        segment_ids=np.asarray(segment_ids, dtype=object),
        crs=gdf.crs,
    )
