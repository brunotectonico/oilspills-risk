"""Lightweight particle-advection module for coastal oil-spill screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
