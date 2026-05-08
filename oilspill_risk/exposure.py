"""Grid-based coastward current exposure calculations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .coastlines import CoastPointSet
from .trajectory import CurrentField, current_field_from_geotiff, current_field_from_netcdf

PERIOD_ID_PATTERN = re.compile(r"_(\S\d+)_*|([DMSY]\d{1,4})_*")


def extract_period_id_from_path(path: Path | str) -> str | None:
    """Extract a seasonal_periods/analysis_periods period id from a filename."""
    match = PERIOD_ID_PATTERN.search(Path(path).name)
    return match.group(1) if match else None


@dataclass(frozen=True)
class CoastwardExposureResult:
    """Per-pixel current-to-coast exposure result."""

    lon: np.ndarray
    lat: np.ndarray
    probability: np.ndarray
    alignment: np.ndarray
    speed: np.ndarray
    distance: np.ndarray
    nearest_coast_x: np.ndarray
    nearest_coast_y: np.ndarray
    period_id: str | None = None


def _grid_coordinates(currents: CurrentField) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D lon/lat coordinate grids for a CurrentField."""
    return np.meshgrid(currents.lon, currents.lat)


def _nearest_points_numpy(
    query_points: np.ndarray,
    target_points: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor lookup with NumPy chunks, used when SciPy is unavailable."""
    nearest = np.empty_like(query_points)
    distances = np.empty(query_points.shape[0], dtype=float)
    for start in range(0, query_points.shape[0], chunk_size):
        stop = min(start + chunk_size, query_points.shape[0])
        deltas = query_points[start:stop, None, :] - target_points[None, :, :]
        dist2 = np.sum(deltas * deltas, axis=2)
        nearest_idx = np.argmin(dist2, axis=1)
        nearest[start:stop] = target_points[nearest_idx]
        distances[start:stop] = np.sqrt(dist2[np.arange(stop - start), nearest_idx])
    return nearest, distances


def nearest_coast_points(
    currents: CurrentField,
    coast_points: CoastPointSet | np.ndarray,
    *,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest coastline point for every current-grid cell."""
    points = coast_points.points if isinstance(coast_points, CoastPointSet) else coast_points
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coast_points must have shape (n_points, 2)")

    lon_grid, lat_grid = _grid_coordinates(currents)
    query_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, nearest_idx = tree.query(query_points)
        nearest = points[nearest_idx]
    except ImportError:
        nearest, distances = _nearest_points_numpy(query_points, points, chunk_size=chunk_size)

    return nearest.reshape((*lon_grid.shape, 2)), distances.reshape(lon_grid.shape)


def coastward_exposure_probability(
    currents: CurrentField,
    coast_points: CoastPointSet | np.ndarray,
    *,
    alpha: float = 1.0,
    speed_reference: float | None = None,
    include_speed: bool = True,
    distance_decay: float | None = None,
    target_points: CoastPointSet | np.ndarray | None = None,
    metric_scaling: bool = True,
    chunk_size: int = 50_000,
    period_id: str | None = None,
) -> CoastwardExposureResult:
    """Compute per-pixel probability that currents point toward coast/targets.

    The directional term is ``max(0, cos(theta)) ** alpha``, where ``theta`` is
    the angle between the current vector and the vector from the grid cell to
    the nearest coast/target point. By default, lon/lat vectors are scaled to
    approximate meters before the dot-product calculation. Optionally multiply
    the directional term by normalized current speed and an exponential
    distance-decay term.
    """
    targets = coast_points if target_points is None else target_points
    lon_grid, lat_grid = _grid_coordinates(currents)
    nearest, _ = nearest_coast_points(currents, targets, chunk_size=chunk_size)

    to_coast_x = nearest[..., 0] - lon_grid
    to_coast_y = nearest[..., 1] - lat_grid
    current_x = currents.u.astype(float)
    current_y = currents.v.astype(float)

    if metric_scaling:
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = np.maximum(1e-6, m_per_deg_lat * np.cos(np.deg2rad(lat_grid)))
        to_coast_x = to_coast_x * m_per_deg_lon
        to_coast_y = to_coast_y * m_per_deg_lat
        current_x = current_x * m_per_deg_lon
        current_y = current_y * m_per_deg_lat

    speed = np.sqrt(current_x**2 + current_y**2)
    coast_distance = np.sqrt(to_coast_x**2 + to_coast_y**2)
    denom = speed * coast_distance

    alignment = np.full_like(speed, np.nan, dtype=float)
    valid = np.isfinite(current_x) & np.isfinite(current_y) & (denom > 0)
    alignment[valid] = (
        current_x[valid] * to_coast_x[valid] + current_y[valid] * to_coast_y[valid]
    ) / denom[valid]
    alignment = np.clip(alignment, -1.0, 1.0)

    probability = np.where(
        np.isfinite(alignment),
        np.maximum(0.0, alignment) ** max(float(alpha), 0.0),
        np.nan,
    )

    if include_speed:
        reference = (
            float(speed_reference)
            if speed_reference is not None
            else float(np.nanpercentile(speed, 95))
        )
        if reference > 0:
            probability = probability * np.clip(speed / reference, 0.0, 1.0)

    if distance_decay is not None and distance_decay > 0:
        probability = probability * np.exp(-coast_distance / float(distance_decay))

    return CoastwardExposureResult(
        lon=currents.lon,
        lat=currents.lat,
        probability=probability.astype("float32"),
        alignment=alignment.astype("float32"),
        speed=speed.astype("float32"),
        distance=coast_distance.astype("float32"),
        nearest_coast_x=nearest[..., 0].astype("float32"),
        nearest_coast_y=nearest[..., 1].astype("float32"),
        period_id=period_id,
    )


def _aggregate_period_id(
    results: list[CoastwardExposureResult],
    period_id: str | None,
) -> str | None:
    """Resolve the period id to attach to an aggregate result."""
    if period_id is not None:
        return period_id

    period_ids = {result.period_id for result in results if result.period_id is not None}
    if len(period_ids) > 1:
        raise ValueError(
            f"Cannot aggregate mixed periods without selecting one period_id: {sorted(period_ids)}"
        )
    return next(iter(period_ids), None)


def aggregate_exposure_probabilities(
    results: list[CoastwardExposureResult],
    *,
    period_id: str | None = None,
) -> CoastwardExposureResult:
    """Average exposure results for one analysis_periods/seasonal_periods label."""
    if not results:
        raise ValueError("At least one exposure result is required")

    aggregate_period_id = _aggregate_period_id(results, period_id)
    if aggregate_period_id is not None:
        results = [
            result
            for result in results
            if result.period_id in {None, aggregate_period_id}
        ]
        if not results:
            raise ValueError(f"No exposure results found for period_id={aggregate_period_id!r}")

    first = results[0]
    probability = np.nanmean(np.stack([result.probability for result in results]), axis=0)
    alignment = np.nanmean(np.stack([result.alignment for result in results]), axis=0)
    speed = np.nanmean(np.stack([result.speed for result in results]), axis=0)
    distance = first.distance

    return CoastwardExposureResult(
        lon=first.lon,
        lat=first.lat,
        probability=probability.astype("float32"),
        alignment=alignment.astype("float32"),
        speed=speed.astype("float32"),
        distance=distance.astype("float32"),
        nearest_coast_x=first.nearest_coast_x,
        nearest_coast_y=first.nearest_coast_y,
        period_id=aggregate_period_id,
    )


def aggregate_exposure_probabilities_by_period(
    results: list[CoastwardExposureResult],
) -> dict[str, CoastwardExposureResult]:
    """Group and average exposure results by repeating period id."""
    grouped: dict[str, list[CoastwardExposureResult]] = {}
    for result in results:
        if result.period_id is None:
            raise ValueError("All exposure results must include period_id for grouped aggregation")
        grouped.setdefault(result.period_id, []).append(result)

    return {
        period_id: aggregate_exposure_probabilities(period_results, period_id=period_id)
        for period_id, period_results in sorted(grouped.items())
    }


def aggregate_exposure_probabilities_for_periods(
    results: list[CoastwardExposureResult],
    periods: list[tuple[date, date, str]],
) -> dict[str, CoastwardExposureResult]:
    """Aggregate exposure results in the order returned by analysis_periods()."""
    by_period = aggregate_exposure_probabilities_by_period(results)
    return {
        period_id: by_period[period_id]
        for _, _, period_id in periods
        if period_id in by_period
    }


def _cell_size(values: np.ndarray) -> float:
    if values.size < 2:
        raise ValueError("At least two coordinates are required to infer a raster cell size")
    return float(np.median(np.diff(np.sort(values))))


def _raster_transform(lon: np.ndarray, lat: np.ndarray):
    xres = _cell_size(lon)
    yres = _cell_size(lat)
    west = float(np.min(lon)) - xres / 2.0
    north = float(np.max(lat)) + yres / 2.0
    return from_origin(west, north, xres, yres)


def _north_up(values: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return values[::-1, :] if lat[0] < lat[-1] else values


def write_exposure_geotiff(
    result: CoastwardExposureResult,
    output_path: Path,
    *,
    layer: str = "probability",
) -> Path:
    """Write an exposure layer to a north-up EPSG:4326 GeoTIFF."""
    data = getattr(result, layer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": _raster_transform(result.lon, result.lat),
        "nodata": np.nan,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(_north_up(data.astype("float32"), result.lat), 1)
    return output_path


def exposure_from_netcdf(
    nc_path: Path,
    coast_points: CoastPointSet | np.ndarray,
    **kwargs,
) -> CoastwardExposureResult:
    """Load a NetCDF current field and compute coastward exposure."""
    period_id = kwargs.pop("period_id", extract_period_id_from_path(nc_path))
    currents = current_field_from_netcdf(nc_path, **kwargs.pop("current_kwargs", {}))
    return coastward_exposure_probability(currents, coast_points, period_id=period_id, **kwargs)


def exposure_from_geotiff(
    u_tif: Path,
    v_tif: Path,
    coast_points: CoastPointSet | np.ndarray,
    **kwargs,
) -> CoastwardExposureResult:
    """Load paired U/V GeoTIFFs and compute coastward exposure."""
    period_id = kwargs.pop("period_id", extract_period_id_from_path(u_tif))
    currents = current_field_from_geotiff(u_tif, v_tif, **kwargs.pop("current_kwargs", {}))
    return coastward_exposure_probability(currents, coast_points, period_id=period_id, **kwargs)
