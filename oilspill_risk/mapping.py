"""Mapping helpers for OSCAR currents, hotspots, and future cartography outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .trajectory import CurrentField, HotspotSource, current_field_from_geotiff, current_field_from_netcdf


def load_current_field(
    *,
    u_tif: Path | None = None,
    v_tif: Path | None = None,
    nc_path: Path | None = None,
    input_units: str = "m/s",
    **netcdf_kwargs: Any,
) -> CurrentField:
    """Load U/V currents from either paired GeoTIFFs or a NetCDF file.

    Use paired GeoTIFFs when matching QGIS/GDAL raster conventions is the
    priority. Use NetCDF when time slices or temporal averaging should be kept
    in the current-loading step.
    """
    if nc_path is not None and (u_tif is not None or v_tif is not None):
        raise ValueError("Choose either nc_path or paired u_tif/v_tif, not both")
    if nc_path is not None:
        return current_field_from_netcdf(nc_path, input_units=input_units, **netcdf_kwargs)
    if u_tif is None or v_tif is None:
        raise ValueError("Provide either nc_path or both u_tif and v_tif")
    return current_field_from_geotiff(u_tif, v_tif, input_units=input_units)


def current_intensity(currents: CurrentField) -> np.ndarray:
    """Return current speed/intensity from U/V components."""
    return np.sqrt(currents.u**2 + currents.v**2)


def add_cartopy_coastlines(
    ax: Any | None = None,
    *,
    resolution: str = "10m",
    linewidth: float = 0.8,
    color: str = "black",
    **coastline_kwargs: Any,
) -> Any:
    """Add Cartopy coastlines to a map axis.

    Cartopy is imported only when this helper is called. If ``ax`` is omitted,
    a PlateCarree GeoAxes is created. This keeps Cartopy optional for users who
    only need basic Matplotlib plots.
    """
    import cartopy.crs as ccrs

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution=resolution, linewidth=linewidth, color=color, **coastline_kwargs)
    return ax


def plot_current_orientation_intensity(
    currents: CurrentField | None = None,
    *,
    u_tif: Path | None = None,
    v_tif: Path | None = None,
    nc_path: Path | None = None,
    input_units: str = "m/s",
    ax: Any | None = None,
    stride: int = 5,
    cmap: str = "viridis",
    quiver_color: str = "white",
    quiver_scale: float | None = None,
    add_colorbar: bool = True,
    title: str | None = "Marine current orientation and intensity",
    add_coastlines: bool = False,
    coastline_resolution: str = "10m",
    **netcdf_kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Plot current intensity as color and orientation as vectors.

    Currents can be passed directly, loaded from paired GeoTIFFs, or loaded
    from NetCDF by passing ``nc_path`` plus any NetCDF loader options such as
    ``time_index`` or ``average_over_time``. Returns ``(ax, mesh, quiver)`` so
    callers can further customize the map.
    """
    if currents is None:
        currents = load_current_field(
            u_tif=u_tif,
            v_tif=v_tif,
            nc_path=nc_path,
            input_units=input_units,
            **netcdf_kwargs,
        )
    if add_coastlines:
        import cartopy.crs as ccrs
        data_transform = ccrs.PlateCarree()
        projection = ccrs.PlateCarree()
    else:
        projection = None
        data_transform = None
        
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})
     
    speed = current_intensity(currents)
    mesh = ax.pcolormesh(currents.lon, currents.lat, speed, 
                         shading="auto", cmap=cmap, 
                         transform=data_transform)

    stride = max(1, int(stride))
    lon_q = currents.lon[::stride]
    lat_q = currents.lat[::stride]
    u_q = currents.u[::stride, ::stride]
    v_q = currents.v[::stride, ::stride]
    quiver = ax.quiver(lon_q, lat_q, u_q, v_q, 
                       color=quiver_color, scale=quiver_scale, 
                       transform=data_transform)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    if add_colorbar:
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Current intensity (degrees/hour after conversion)")
    if add_coastlines:
        add_cartopy_coastlines(ax=ax, resolution=coastline_resolution)
        
    return ax, mesh, quiver


def plot_current_orientation_intensity_from_netcdf(
    nc_path: Path,
    **plot_kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Convenience wrapper to map current intensity/orientation directly from NetCDF."""
    return plot_current_orientation_intensity(nc_path=nc_path, **plot_kwargs)


def _hotspots_to_arrays(
    hotspots: Path | Iterable[HotspotSource] | Any,
    *,
    lon_col: str,
    lat_col: str,
    density_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str | None]]:
    if isinstance(hotspots, (Path, str)):
        import pandas as pd
        hotspots = pd.read_csv(hotspots)

    if hasattr(hotspots, "__getitem__") and all(col in hotspots for col in (lon_col, lat_col, density_col)):
        lon = np.asarray(hotspots[lon_col], dtype=float)
        lat = np.asarray(hotspots[lat_col], dtype=float)
        density = np.asarray(hotspots[density_col], dtype=float)
        labels = list(hotspots["hotspot_id"]) if "hotspot_id" in hotspots else [None] * lon.size
        return lon, lat, density, labels

    lon_values: list[float] = []
    lat_values: list[float] = []
    density_values: list[float] = []
    labels: list[str | None] = []
    for item in hotspots:
        if isinstance(item, HotspotSource):
            lon_values.append(float(item.lon))
            lat_values.append(float(item.lat))
            density_values.append(float(item.density_weight))
            labels.append(str(item.hotspot_id))
        else:
            lon_values.append(float(item[lon_col]))
            lat_values.append(float(item[lat_col]))
            density_values.append(float(item[density_col]))
            labels.append(str(item.get("hotspot_id")) if hasattr(item, "get") and item.get("hotspot_id") else None)

    return (
        np.asarray(lon_values, dtype=float),
        np.asarray(lat_values, dtype=float),
        np.asarray(density_values, dtype=float),
        labels,
    )


def hotspot_symbol_sizes(
    density: np.ndarray,
    *,
    min_size: float = 20.0,
    max_size: float = 250.0,
) -> np.ndarray:
    """Scale density values to Matplotlib marker sizes."""
    density = np.asarray(density, dtype=float)
    if density.size == 0:
        return density
    dmin = float(np.nanmin(density))
    dmax = float(np.nanmax(density))
    if np.isclose(dmin, dmax):
        return np.full_like(density, (min_size + max_size) / 2.0, dtype=float)
    normalized = (density - dmin) / (dmax - dmin)
    return min_size + normalized * (max_size - min_size)


def plot_hotspots(
    hotspots: Path | Iterable[HotspotSource] | Any,
    *,
    ax: Any | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    density_col: str = "density_weight",
    min_size: float = 20.0,
    max_size: float = 250.0,
    color: str = "crimson",
    edgecolor: str = "black",
    alpha: float = 0.75,
    annotate: bool = False,
    label: str = "Traffic-density hotspots",
    **scatter_kwargs: Any,
) -> tuple[Any, Any]:
    """Plot traffic-density hotspots with marker size scaled by density."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    lon, lat, density, labels = _hotspots_to_arrays(
        hotspots,
        lon_col=lon_col,
        lat_col=lat_col,
        density_col=density_col,
    )
    sizes = hotspot_symbol_sizes(density, min_size=min_size, max_size=max_size)
    scatter = ax.scatter(
        lon,
        lat,
        s=sizes,
        c=color,
        edgecolors=edgecolor,
        alpha=alpha,
        label=label,
        **scatter_kwargs,
    )

    if annotate:
        for x, y, text in zip(lon, lat, labels, strict=False):
            if text:
                ax.annotate(text, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return ax, scatter
