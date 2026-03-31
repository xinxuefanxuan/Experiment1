"""Canonical UV Position Map generation.

This module implements step A from the design:
- build a canonical UV position map with shape (H, W, 3)
- each valid UV pixel stores a 3D point interpolated from FLAME vertices
- invalid/background pixels are filled with a fixed value (default: 0)

Only NumPy is required for the core logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class MeshData:
    """Minimal mesh data needed for canonical UV map generation.

    Attributes:
        vertices: 3D vertices in canonical/object space, shape (N, 3).
        faces: triangle indices for `vertices`, shape (M, 3), int.
        uv_vertices: UV coordinates in [0, 1], shape (Nt, 2).
        uv_faces: triangle indices for `uv_vertices`, shape (M, 3), int.
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv_vertices: np.ndarray
    uv_faces: np.ndarray


@dataclass
class RasterResult:
    """Rasterization outputs in UV space."""

    pix2face: np.ndarray  # (H, W), -1 means background
    barycentric: np.ndarray  # (H, W, 3), valid only when pix2face >= 0
    depth: np.ndarray  # (H, W), for tie-breaking in overlap regions


EPS = 1e-8


def _to_pixel_uv(uv: np.ndarray, h: int, w: int, flip_v: bool) -> np.ndarray:
    """Convert UV coordinates [0,1] to pixel coordinates [0,w-1]/[0,h-1]."""
    uv = np.asarray(uv, dtype=np.float64)
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    if flip_v:
        v = 1.0 - v

    x = u * (w - 1)
    y = v * (h - 1)
    return np.stack([x, y], axis=1)


def _barycentric_coords(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Compute barycentric coordinates of p wrt triangle (a,b,c)."""
    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c

    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay

    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y
    d20 = v2x * v0x + v2y * v0y
    d21 = v2x * v1x + v2y * v1y

    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPS:
        return np.nan, np.nan, np.nan

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def rasterize_uv(
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    resolution: int = 1024,
    flip_v: bool = True,
) -> RasterResult:
    """Rasterize UV triangles to get pix2face and barycentric coordinates.

    Note:
        This is a CPU NumPy reference implementation.
        It is straightforward and easy to debug, but not optimized.
    """
    h = w = int(resolution)
    uv_px = _to_pixel_uv(uv_vertices, h=h, w=w, flip_v=flip_v)

    pix2face = np.full((h, w), -1, dtype=np.int32)
    bary = np.zeros((h, w, 3), dtype=np.float32)
    # Canonical map doesn't have true camera depth; we use triangle index tie-break.
    depth = np.full((h, w), np.inf, dtype=np.float32)

    for fi, tri in enumerate(uv_faces):
        ia, ib, ic = int(tri[0]), int(tri[1]), int(tri[2])
        a = uv_px[ia]
        b = uv_px[ib]
        c = uv_px[ic]

        min_x = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
        max_x = min(int(np.ceil(max(a[0], b[0], c[0]))), w - 1)
        min_y = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
        max_y = min(int(np.ceil(max(a[1], b[1], c[1]))), h - 1)

        if min_x > max_x or min_y > max_y:
            continue

        # tie-break priority (lower is better), deterministic for overlaps
        tri_priority = float(fi)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # sample at pixel center
                w0, w1, w2 = _barycentric_coords(
                    p=(x + 0.5, y + 0.5),
                    a=(a[0], a[1]),
                    b=(b[0], b[1]),
                    c=(c[0], c[1]),
                )

                if np.isnan(w0):
                    continue

                if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                    if tri_priority < depth[y, x]:
                        depth[y, x] = tri_priority
                        pix2face[y, x] = fi
                        bary[y, x, 0] = w0
                        bary[y, x, 1] = w1
                        bary[y, x, 2] = w2

    return RasterResult(pix2face=pix2face, barycentric=bary, depth=depth)


def interpolate_uv_position_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    pix2face: np.ndarray,
    barycentric: np.ndarray,
    background_value: float = 0.0,
) -> np.ndarray:
    """Interpolate 3D positions for each UV pixel from mesh vertices.

    Returns:
        uv_position_map: (H, W, 3)
    """
    h, w = pix2face.shape
    uv_pos = np.full((h, w, 3), background_value, dtype=np.float32)

    valid = pix2face >= 0
    ys, xs = np.where(valid)

    for y, x in zip(ys, xs):
        fi = int(pix2face[y, x])
        i0, i1, i2 = faces[fi]
        w0, w1, w2 = barycentric[y, x]
        uv_pos[y, x] = (
            w0 * vertices[int(i0)]
            + w1 * vertices[int(i1)]
            + w2 * vertices[int(i2)]
        )

    return uv_pos


def normalize_for_visualization(uv_pos: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Normalize xyz channels to [0, 255] for visualization PNG."""
    vis = np.zeros_like(uv_pos, dtype=np.float32)

    if not np.any(valid_mask):
        return vis.astype(np.uint8)

    valid_vals = uv_pos[valid_mask]
    mn = valid_vals.min(axis=0)
    mx = valid_vals.max(axis=0)
    scale = np.maximum(mx - mn, 1e-8)

    vis[valid_mask] = (uv_pos[valid_mask] - mn) / scale
    vis = np.clip(vis * 255.0, 0.0, 255.0)
    return vis.astype(np.uint8)


def validate_mesh_data(mesh: MeshData) -> None:
    """Basic shape/range checks to fail early with useful messages."""
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {mesh.vertices.shape}")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"faces must be (M,3), got {mesh.faces.shape}")
    if mesh.uv_vertices.ndim != 2 or mesh.uv_vertices.shape[1] != 2:
        raise ValueError(
            f"uv_vertices must be (Nt,2), got {mesh.uv_vertices.shape}"
        )
    if mesh.uv_faces.ndim != 2 or mesh.uv_faces.shape[1] != 3:
        raise ValueError(f"uv_faces must be (M,3), got {mesh.uv_faces.shape}")

    if mesh.faces.shape[0] != mesh.uv_faces.shape[0]:
        raise ValueError(
            "faces and uv_faces must have same triangle count, got "
            f"{mesh.faces.shape[0]} vs {mesh.uv_faces.shape[0]}"
        )

    n_v = mesh.vertices.shape[0]
    n_vt = mesh.uv_vertices.shape[0]

    if mesh.faces.min() < 0 or mesh.faces.max() >= n_v:
        raise ValueError("faces has index out of range for vertices")
    if mesh.uv_faces.min() < 0 or mesh.uv_faces.max() >= n_vt:
        raise ValueError("uv_faces has index out of range for uv_vertices")


def load_mesh_data(
    vertices_path: Path,
    faces_path: Path,
    uv_vertices_path: Path,
    uv_faces_path: Path,
) -> MeshData:
    """Load mesh arrays from .npy files.

    Required input format:
    - vertices.npy: (N,3), float32/float64
    - faces.npy: (M,3), int32/int64
    - uv_vertices.npy: (Nt,2), float32/float64 in [0,1]
    - uv_faces.npy: (M,3), int32/int64
    """
    vertices = np.load(vertices_path)
    faces = np.load(faces_path)
    uv_vertices = np.load(uv_vertices_path)
    uv_faces = np.load(uv_faces_path)

    mesh = MeshData(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        uv_vertices=np.asarray(uv_vertices, dtype=np.float32),
        uv_faces=np.asarray(uv_faces, dtype=np.int32),
    )
    validate_mesh_data(mesh)
    return mesh


def generate_canonical_uv_position_map(
    mesh: MeshData,
    resolution: int = 1024,
    flip_v: bool = True,
    background_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Main pipeline for Step-A canonical UV position map.

    Returns:
        uv_position_map: (H, W, 3), float32
        uv_valid_mask: (H, W), bool
        pix2face: (H, W), int32
        barycentric: (H, W, 3), float32
    """
    validate_mesh_data(mesh)

    rast = rasterize_uv(
        uv_vertices=mesh.uv_vertices,
        uv_faces=mesh.uv_faces,
        resolution=resolution,
        flip_v=flip_v,
    )
    uv_pos = interpolate_uv_position_map(
        vertices=mesh.vertices,
        faces=mesh.faces,
        pix2face=rast.pix2face,
        barycentric=rast.barycentric,
        background_value=background_value,
    )
    valid_mask = rast.pix2face >= 0

    return uv_pos, valid_mask, rast.pix2face, rast.barycentric
