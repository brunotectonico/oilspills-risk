"""Mean-density raster aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

import logging # Added to debug
LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class RasterGroup:
    """Defines output grouping for mean-density rasters."""

    key: str
    filename: str


class MeanRasterAggregator:
    """Accumulate and write mean rasters for arbitrary grouping keys."""

    def __init__(self) -> None:
        self._sum: dict[str, np.ndarray] = {}
        self._count: dict[str, np.ndarray] = {}
        self._transform = None

    def add(self, group: RasterGroup, density: np.ndarray, transform: rasterio.Affine) -> None:
        """Accumulate one density raster into the requested group."""
        if self._transform is None:
            self._transform = transform

        if group.key not in self._sum:
            self._sum[group.key] = np.zeros_like(density, dtype=float)
            self._count[group.key] = np.zeros_like(density, dtype=float)

        valid = ~np.isnan(density)
        self._sum[group.key][valid] += density[valid]
        self._count[group.key][valid] += 1

    def write_all(self, output_dir: Path) -> list[Path]:
        """Write all grouped mean rasters to GeoTIFF files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for key in sorted(self._sum):
            # mean_density = np.where(
            #     self._count[key] > 0,
            #     self._sum[key] / self._count[key],
            #     np.nan,
            # ).astype("float32")
            count = self._count[key]
            if not np.any(count > 0):
                LOGGER.warning("Skipping empty group %s", key)
                continue
            mean_density = np.full_like(self._sum[key], np.nan, dtype="float32")
            np.divide(self._sum[key], count, out=mean_density, where=count > 0)

            out_path = output_dir / f"mean_density_{key}.tif"
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=mean_density.shape[0],
                width=mean_density.shape[1],
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=self._transform,
                nodata=np.nan,
            ) as dst:
                dst.write(mean_density, 1)
            written.append(out_path)

        return written
