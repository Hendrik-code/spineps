"""Resolution-aware thresholding: convert physical (mm) thresholds to voxels at the actual image zoom.

SPINEPS was originally tuned on T2w MR images, so several processing thresholds were hard-coded as
voxel counts that implicitly assumed the model resolution. To support inputs at other resolutions
(e.g. CT), those thresholds are now expressed in physical millimetres and converted back to voxels at
runtime using the zoom of the image being processed. The constants are derived from
``REFERENCE_ZOOM`` so that, at the reference resolution, the converted voxel counts equal the original
hard-coded values (i.e. existing T2w results are preserved exactly).
"""

from __future__ import annotations

import numpy as np
from TPTBox import ZOOMS

# Canonical voxel spacing (mm) of the SPINEPS T2w semantic and instance models
# (sagittal: 0.75 mm in-plane, 1.65 mm superior-inferior). All physical thresholds in the pipeline
# are derived from this reference so behaviour is unchanged at this resolution.
REFERENCE_ZOOM: ZOOMS = (0.75, 0.75, 1.65)

# Physical volume (mm^3) of a single voxel at the reference resolution.
REFERENCE_VOXEL_VOLUME_MM3: float = float(np.prod(REFERENCE_ZOOM))

# In a ``P, I, R`` oriented image, these are the in-plane (axial) axes; the remaining axis (1) is the
# superior-inferior height axis.
INFERIOR_AXIS_PIR: int = 1
INPLANE_AXES_PIR: tuple[int, int] = (0, 2)


def mm3_to_voxels(threshold_mm3: float, zoom: ZOOMS, minimum: int = 1) -> int:
    """Convert a volume threshold in mm^3 to a voxel count for the given voxel spacing.

    Args:
        threshold_mm3 (float): Volume threshold in cubic millimetres.
        zoom (ZOOMS): Voxel spacing (mm) of the image being processed, per axis.
        minimum (int, optional): Lower bound on the returned voxel count. Defaults to 1.

    Returns:
        int: The threshold expressed as a number of voxels at ``zoom`` (at least ``minimum``).
    """
    voxel_volume = float(np.prod(zoom))
    return max(round(threshold_mm3 / voxel_volume), minimum)


def mm2_to_voxels(threshold_mm2: float, zoom: ZOOMS, plane_axes: tuple[int, int] = INPLANE_AXES_PIR, minimum: int = 1) -> int:
    """Convert an in-plane area threshold in mm^2 to a voxel count for the given voxel spacing.

    Args:
        threshold_mm2 (float): Area threshold in square millimetres.
        zoom (ZOOMS): Voxel spacing (mm) of the image being processed, per axis.
        plane_axes (tuple[int, int], optional): The two axes spanning the plane of interest.
            Defaults to the in-plane (axial) axes of a P,I,R image.
        minimum (int, optional): Lower bound on the returned voxel count. Defaults to 1.

    Returns:
        int: The threshold expressed as a number of voxels at ``zoom`` (at least ``minimum``).
    """
    voxel_area = float(zoom[plane_axes[0]] * zoom[plane_axes[1]])
    return max(round(threshold_mm2 / voxel_area), minimum)


def mm_to_voxels(threshold_mm: float, zoom: ZOOMS, minimum: int = 0) -> int:
    """Convert a distance threshold in mm to an (isotropic) voxel count using the finest spacing.

    Uses the smallest voxel spacing so the result matches the original voxel-isotropic behaviour at
    the reference resolution; this is the same convention used for the labeling crop margin.

    Args:
        threshold_mm (float): Distance threshold in millimetres.
        zoom (ZOOMS): Voxel spacing (mm) of the image being processed, per axis.
        minimum (int, optional): Lower bound on the returned voxel count. Defaults to 0.

    Returns:
        int: The threshold expressed as a number of voxels at ``zoom`` (at least ``minimum``).
    """
    return max(round(threshold_mm / float(min(zoom))), minimum)


def mm_to_voxels_axis(threshold_mm: float, zoom: ZOOMS, axis: int) -> float:
    """Convert a distance threshold in mm to a (possibly fractional) voxel distance along one axis.

    Returns a float so it can be passed straight to ``NII.compute_crop(dist=...)``.

    Args:
        threshold_mm (float): Distance threshold in millimetres.
        zoom (ZOOMS): Voxel spacing (mm) of the image being processed, per axis.
        axis (int): The axis along which the distance is measured.

    Returns:
        float: The threshold expressed as a voxel distance along ``axis`` at ``zoom``.
    """
    return threshold_mm / float(zoom[axis])
