"""OSCAR grid standardization, clipping, and GIS export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from .models import StudyArea, validate_study_area

DEFAULT_U_NAMES = ("u", "u_current", "ugos", "u_curr")
DEFAULT_V_NAMES = ("v", "v_current", "vgos", "v_curr")


def _infer_lon_lat_names(ds: xr.Dataset, lon_name: str = "lon", lat_name: str = "lat") -> tuple[str, str]:
    """Resolve longitude/latitude names from common metadata and variable names."""

    def _is_lon(var_key: str, var_obj: xr.DataArray | xr.Variable) -> bool:
        attrs = getattr(var_obj, "attrs", {})
        std = str(attrs.get("standard_name", "")).lower()
        units = str(attrs.get("units", "")).lower()
        axis = str(attrs.get("axis", "")).upper()
        name = str(var_key).lower()
        return axis == "X" or "longitude" in std or "degrees_east" in units or name in {"lon", "longitude", "x"}

    def _is_lat(var_key: str, var_obj: xr.DataArray | xr.Variable) -> bool:
        attrs = getattr(var_obj, "attrs", {})
        std = str(attrs.get("standard_name", "")).lower()
        units = str(attrs.get("units", "")).lower()
        axis = str(attrs.get("axis", "")).upper()
        name = str(var_key).lower()
        return axis == "Y" or "latitude" in std or "degrees_north" in units or name in {"lat", "latitude", "y"}

    merged = {**dict(ds.coords.items()), **dict(ds.variables.items())}
    lon_found = [name for name, obj in merged.items() if _is_lon(name, obj)]
    lat_found = [name for name, obj in merged.items() if _is_lat(name, obj)]

    resolved_lon = lon_found[0] if lon_found else lon_name
    resolved_lat = lat_found[0] if lat_found else lat_name

    if resolved_lon == resolved_lat:
        if lon_name in ds.variables or lon_name in ds.coords or lon_name in ds.dims:
            resolved_lon = lon_name
        if lat_name in ds.variables or lat_name in ds.coords or lat_name in ds.dims:
            resolved_lat = lat_name
    if resolved_lon == resolved_lat:
        raise ValueError(f"Could not reliably infer lon/lat coordinate names (got {resolved_lon})")

    return resolved_lon, resolved_lat


def _metadata_float(ds: xr.Dataset, name: str, default: float | None = None) -> float | None:
    value = ds.attrs.get(name, default)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split()[0]
    return float(value)


def _reconstruct_degree_coords(ds: xr.Dataset, lon_name: str, lat_name: str) -> xr.Dataset:
    """Convert index-like lon/lat coordinates to degree coordinates using CF metadata."""
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lat = np.asarray(ds[lat_name].values, dtype=float)

    if np.allclose(lon, np.arange(lon.size)):
        lon_min = _metadata_float(ds, "geospatial_lon_min")
        lon_res = _metadata_float(ds, "geospatial_lon_resolution", 0.25)
        if lon_min is None:
            raise ValueError("Missing geospatial_lon_min for reconstructing longitude")
        ds = ds.assign_coords({lon_name: lon_min + np.arange(lon.size) * lon_res})

    if np.allclose(lat, np.arange(lat.size)):
        lat_min = _metadata_float(ds, "geospatial_lat_min")
        lat_res = _metadata_float(ds, "geospatial_lat_resolution", 0.25)
        if lat_min is None:
            raise ValueError("Missing geospatial_lat_min for reconstructing latitude")
        ds = ds.assign_coords({lat_name: lat_min + np.arange(lat.size) * lat_res})

    return ds


def ensure_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    """Return a dataset with canonical ``lon``/``lat`` dimensions and coordinates.

    OSCAR files may store spatial axes as dimensions named ``longitude`` and
    ``latitude`` while the coordinate variables are named ``lon`` and ``lat``
    (or vice versa).  Canonicalizing both the coordinate names and the backing
    dimension names avoids downstream errors when arrays are transposed for
    raster export.
    """
    lon_name, lat_name = _infer_lon_lat_names(ds)
    ds = _reconstruct_degree_coords(ds, lon_name=lon_name, lat_name=lat_name)

    lon_dim = ds[lon_name].dims[0]
    lat_dim = ds[lat_name].dims[0]
    rename: dict[str, str] = {}
    for source, target in (
        (lon_dim, "lon"),
        (lat_dim, "lat"),
        (lon_name, "lon"),
        (lat_name, "lat"),
    ):
        if source != target and source in ds:
            rename[source] = target
        elif source != target and source in ds.dims:
            rename[source] = target

    if rename:
        ds = ds.rename(rename)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        ds = ds.set_coords([name for name in ("lon", "lat") if name in ds.variables])

    if float(ds["lon"].max()) > 180.0:
        ds = ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0)

    return ds.sortby("lat").sortby("lon")


def subset_lon_lat(ds: xr.Dataset, area: StudyArea, lon_name: str = "lon", lat_name: str = "lat") -> xr.Dataset:
    """Subset a rectilinear lon/lat grid to a study area, including dateline wrap."""
    area = validate_study_area(area)
    lon_vals = np.asarray(ds[lon_name].values, dtype=float)
    lat_vals = np.asarray(ds[lat_name].values, dtype=float)

    lon_min = area.lon_min
    lon_max = area.lon_max
    if lon_vals.min() >= 0 and lon_min < 0:
        lon_min %= 360.0
    if lon_vals.min() >= 0 and lon_max < 0:
        lon_max %= 360.0

    if lon_min <= lon_max:
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    else:
        lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)
    lat_mask = (lat_vals >= area.lat_min) & (lat_vals <= area.lat_max)

    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    if lon_idx.size == 0 or lat_idx.size == 0:
        raise ValueError(f"No OSCAR grid cells found within requested bounds: {area}")

    return ds.isel({ds[lon_name].dims[0]: lon_idx, ds[lat_name].dims[0]: lat_idx})


def _pick_uv_names(ds: xr.Dataset, u_var: str = "u", v_var: str = "v") -> tuple[str, str]:
    u_candidates = (u_var, *[name for name in DEFAULT_U_NAMES if name != u_var])
    v_candidates = (v_var, *[name for name in DEFAULT_V_NAMES if name != v_var])
    actual_u = next((name for name in u_candidates if name in ds.data_vars), None)
    actual_v = next((name for name in v_candidates if name in ds.data_vars), None)
    if actual_u is None or actual_v is None:
        raise KeyError(f"No U/V variables found. Variables: {list(ds.data_vars)}")
    return actual_u, actual_v


def _clean_fill_values(ds: xr.Dataset, variable_names: tuple[str, str]) -> xr.Dataset:
    for name in variable_names:
        fill_value = ds[name].attrs.get("_FillValue")
        if fill_value is not None:
            ds[name] = ds[name].where(ds[name] != fill_value)
    return ds


def _as_qgis_netcdf_grid(ds: xr.Dataset) -> xr.Dataset:
    """Order NetCDF variables as non-spatial dims, then lat/lon, north-to-south.

    QGIS is more reliable with NetCDF rasters when spatial dimensions are in
    the conventional Y/X order.  Keeping latitude descending makes the NetCDF
    array north-up, matching the GeoTIFF export without changing the GeoTIFF
    workflow itself.
    """
    ds = ds.sortby("lon")
    ds = ds.sortby("lat", ascending=False)

    ordered_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if "lat" in da.dims and "lon" in da.dims:
            leading_dims = [dim for dim in da.dims if dim not in {"lat", "lon"}]
            ordered_vars[name] = da.transpose(*leading_dims, "lat", "lon")
        else:
            ordered_vars[name] = da

    if not ordered_vars:
        return ds
    return ds.assign(ordered_vars)


def standardize_oscar_uv_netcdf(
    input_nc: Path,
    output_nc: Path,
    area: StudyArea | None = None,
    *,
    u_var: str = "u",
    v_var: str = "v",
) -> Path:
    """Normalize OSCAR lon/lat coordinates and optionally clip to a study area.

    Pass ``area=None`` to keep the full OSCAR domain.  This is useful for
    checking whether coordinate rearrangement fixes the global placement before
    testing a local Red Sea/Djibouti clip.
    """
    with xr.open_dataset(input_nc) as ds:
        ds = ensure_lon_lat(ds)
        actual_u, actual_v = _pick_uv_names(ds, u_var=u_var, v_var=v_var)
        output = ds[[actual_u, actual_v]]
        if area is not None:
            output = subset_lon_lat(output, area)
        output = _clean_fill_values(output, (actual_u, actual_v))

        output = _as_qgis_netcdf_grid(output)
        output["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
        output["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})
        output.attrs.update(
            geospatial_lon_min=float(output["lon"].min()),
            geospatial_lon_max=float(output["lon"].max()),
            geospatial_lat_min=float(output["lat"].min()),
            geospatial_lat_max=float(output["lat"].max()),
        )

        output_nc.parent.mkdir(parents=True, exist_ok=True)
        output.to_netcdf(output_nc)
    return output_nc


def _cell_size(values: np.ndarray) -> float:
    if values.size < 2:
        raise ValueError("At least two coordinates are required to infer a raster cell size")
    return float(np.median(np.diff(np.sort(values))))


def _north_up_array(da: xr.DataArray) -> np.ndarray:
    """Return a 2-D array ordered north-to-south and west-to-east."""
    lat_dim = "lat" if "lat" in da.dims else da["lat"].dims[0]
    lon_dim = "lon" if "lon" in da.dims else da["lon"].dims[0]
    extra_dims = [dim for dim in da.dims if dim not in {lat_dim, lon_dim}]
    for dim in extra_dims:
        if da.sizes[dim] != 1:
            raise ValueError(f"Cannot export non-spatial dimension {dim!r} with size {da.sizes[dim]}")
        da = da.isel({dim: 0}, drop=True)

    lat_values = np.asarray(da["lat"].values, dtype=float)
    values = da.transpose(lat_dim, lon_dim).values.astype("float32")
    return values[::-1, :] if lat_values[0] < lat_values[-1] else values


def _raster_transform(lon: np.ndarray, lat: np.ndarray):
    """Create a north-up transform from coordinate centers, not center bounds."""
    xres = _cell_size(lon)
    yres = _cell_size(lat)
    west = float(np.min(lon)) - xres / 2.0
    north = float(np.max(lat)) + yres / 2.0
    return from_origin(west, north, xres, yres)


def export_oscar_uv_geotiff(
    input_nc: Path,
    output_dir: Path,
    area: StudyArea | None = None,
    *,
    u_var: str = "u",
    v_var: str = "v",
) -> tuple[Path, Path]:
    """Export OSCAR U/V currents as north-up EPSG:4326 GeoTIFF rasters.

    Pass ``area=None`` to export the full standardized domain.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_nc) as ds:
        ds = ensure_lon_lat(ds)
        actual_u, actual_v = _pick_uv_names(ds, u_var=u_var, v_var=v_var)
        output = ds[[actual_u, actual_v]]
        if area is not None:
            output = subset_lon_lat(output, area)
        output = _clean_fill_values(output, (actual_u, actual_v))

        u = output[actual_u].isel(time=0) if "time" in output[actual_u].dims else output[actual_u]
        v = output[actual_v].isel(time=0) if "time" in output[actual_v].dims else output[actual_v]
        lon = np.asarray(output["lon"].values, dtype=float)
        lat = np.asarray(output["lat"].values, dtype=float)
        transform = _raster_transform(lon, lat)
        height, width = lat.size, lon.size

        u_path = output_dir / f"{input_nc.stem}_{u_var}.tif"
        v_path = output_dir / f"{input_nc.stem}_{v_var}.tif"
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
        }

        with rasterio.open(u_path, "w", **profile) as dst:
            dst.write(_north_up_array(u), 1)
        with rasterio.open(v_path, "w", **profile) as dst:
            dst.write(_north_up_array(v), 1)

    return u_path, v_path
