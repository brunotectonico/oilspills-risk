"""Lightweight particle-advection module for coastal oil-spill screening."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
import xarray as xr


@dataclass(frozen=True)
class HotspotSource:
    """Potential spill origin derived from traffic-density hotspots."""

    lon: float
    lat: float
    density_weight: float
    hotspot_id: str


@dataclass(frozen=True)
class SimulationConfig:
    """Controls a minimal trajectory simulation."""

    n_particles: int = 250
    horizon_hours: int = 24 * 7
    dt_hours: int = 1
    diffusion_deg_per_sqrt_hour: float = 0.01
    daily_mass_loss_fraction: float = 0.30
    coastal_buffer_deg: float = 0.10


@dataclass(frozen=True)
class CoastalRiskResult:
    """Risk summary for one hotspot source."""

    hotspot_id: str
    density_factor: float
    coastal_hit_fraction: float
    survival_fraction: float
    probability_score: float


@dataclass(frozen=True)
class CurrentField:
    """Regular-grid currents. Units for u/v are degrees per hour."""

    lon: np.ndarray
    lat: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def velocity_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbor velocity lookup at particle locations."""
        xi = np.abs(self.lon[None, :] - x[:, None]).argmin(axis=1)
        yi = np.abs(self.lat[None, :] - y[:, None]).argmin(axis=1)
        return self.u[yi, xi], self.v[yi, xi]


def _convert_current_units(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    input_units: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert U/V current components to degrees per hour for trajectory steps."""
    if input_units != "m/s":
        return u, v

    m_per_deg_lat = 111_320.0
    lat_rad = np.deg2rad(lat)
    m_per_deg_lon = np.maximum(1e-6, m_per_deg_lat * np.cos(lat_rad))

    v = v * 3600.0 / m_per_deg_lat
    u = u * 3600.0 / m_per_deg_lon[:, None]
    return u, v


def current_field_from_geotiff(
    u_tif: Path,
    v_tif: Path,
    *,
    input_units: str = "m/s",
) -> CurrentField:
    """Build a CurrentField from paired U/V GeoTIFF rasters.

    This is useful when the OSCAR rasters have already been validated in QGIS
    or need to align with QGIS-style marine-traffic density raster products.
    The U and V rasters must share shape, transform, and CRS.
    """
    with rasterio.open(u_tif) as u_src, rasterio.open(v_tif) as v_src:
        if u_src.shape != v_src.shape:
            raise ValueError(f"U/V raster shapes differ: {u_src.shape} != {v_src.shape}")
        if u_src.transform != v_src.transform:
            raise ValueError("U/V raster transforms differ")
        if u_src.crs != v_src.crs:
            raise ValueError(f"U/V raster CRS differ: {u_src.crs} != {v_src.crs}")

        u = u_src.read(1, masked=True).filled(np.nan).astype(float)
        v = v_src.read(1, masked=True).filled(np.nan).astype(float)
        height, width = u_src.height, u_src.width
        cols = np.arange(width)
        rows = np.arange(height)
        lon = np.asarray(xy(u_src.transform, np.zeros(width, dtype=int), cols, offset="center")[0], dtype=float)
        lat = np.asarray(xy(u_src.transform, rows, np.zeros(height, dtype=int), offset="center")[1], dtype=float)

    u, v = _convert_current_units(lon=lon, lat=lat, u=u, v=v, input_units=input_units)
    return CurrentField(lon=lon, lat=lat, u=u, v=v)


def current_field_from_netcdf(
    nc_path: Path,
    *,
    u_var: str = "u",
    v_var: str = "v",
    lon_name: str = "lon",
    lat_name: str = "lat",
    time_name: str = "time",
    time_index: int | None = None,
    average_over_time: bool = True,
    input_units: str = "m/s",
) -> CurrentField:
    """Build a CurrentField from u/v variables in NetCDF.

    The function accepts u/v on (time, lat, lon) or (lat, lon).
    If units are m/s, conversion to deg/hour is approximated with local latitude.
    """
    ds = xr.open_dataset(nc_path)
    u_da = ds[u_var]
    v_da = ds[v_var]

    if time_name in u_da.dims:
        if time_index is not None:
            u_slice = u_da.isel({time_name: time_index})
            v_slice = v_da.isel({time_name: time_index})
        elif average_over_time:
            u_slice = u_da.mean(dim=time_name)
            v_slice = v_da.mean(dim=time_name)
        else:
            u_slice = u_da.isel({time_name: 0})
            v_slice = v_da.isel({time_name: 0})
    else:
        u_slice = u_da
        v_slice = v_da

    lon = u_slice[lon_name].values.astype(float)
    lat = u_slice[lat_name].values.astype(float)
    u = u_slice.values.astype(float)
    v = v_slice.values.astype(float)

    u, v = _convert_current_units(lon=lon, lat=lat, u=u, v=v, input_units=input_units)

    ds.close()
    return CurrentField(lon=lon, lat=lat, u=u, v=v)


def _mass_survival_fraction(hours: int, daily_mass_loss_fraction: float) -> float:
    hours = max(hours, 0)
    daily_survival = max(0.0, min(1.0, 1.0 - daily_mass_loss_fraction))
    return daily_survival ** (hours / 24.0)


def simulate_particles(
    source: HotspotSource,
    currents: CurrentField,
    cfg: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate particle positions with advection + random walk diffusion."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_steps = max(1, cfg.horizon_hours // cfg.dt_hours)
    x = np.full(cfg.n_particles, source.lon, dtype=float)
    y = np.full(cfg.n_particles, source.lat, dtype=float)

    for _ in range(n_steps):
        u, v = currents.velocity_at(x, y)
        x += u * cfg.dt_hours
        y += v * cfg.dt_hours

        sigma = cfg.diffusion_deg_per_sqrt_hour * np.sqrt(cfg.dt_hours)
        x += rng.normal(0.0, sigma, size=cfg.n_particles)
        y += rng.normal(0.0, sigma, size=cfg.n_particles)

    return x, y


def estimate_coastal_risk(
    source: HotspotSource,
    currents: CurrentField,
    coast_points: np.ndarray,
    cfg: SimulationConfig,
) -> CoastalRiskResult:
    """Combine density weighting with coastal-hit fraction and weathering survival."""
    x_end, y_end = simulate_particles(source=source, currents=currents, cfg=cfg)

    particle_points = np.column_stack((x_end, y_end))
    deltas = particle_points[:, None, :] - coast_points[None, :, :]
    nearest_dist = np.sqrt((deltas**2).sum(axis=2)).min(axis=1)

    coastal_hits = nearest_dist <= cfg.coastal_buffer_deg
    coastal_hit_fraction = float(np.mean(coastal_hits))
    survival_fraction = _mass_survival_fraction(cfg.horizon_hours, cfg.daily_mass_loss_fraction)

    probability_score = float(source.density_weight * coastal_hit_fraction * survival_fraction)

    return CoastalRiskResult(
        hotspot_id=source.hotspot_id,
        density_factor=float(source.density_weight),
        coastal_hit_fraction=coastal_hit_fraction,
        survival_fraction=survival_fraction,
        probability_score=probability_score,
    )