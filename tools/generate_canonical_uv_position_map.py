#!/usr/bin/env python3
"""CLI: Generate canonical UV position map (Step A)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from uv_position_map import (
    generate_canonical_uv_position_map,
    load_mesh_data,
    normalize_for_visualization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical UV position map from mesh + UV topology."
    )
    parser.add_argument("--vertices", type=Path, required=True, help="Path to vertices.npy")
    parser.add_argument("--faces", type=Path, required=True, help="Path to faces.npy")
    parser.add_argument(
        "--uv-vertices", type=Path, required=True, help="Path to uv_vertices.npy"
    )
    parser.add_argument("--uv-faces", type=Path, required=True, help="Path to uv_faces.npy")

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Output UV resolution H=W (default: 1024)",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=0.0,
        help="Background value for invalid UV pixels (default: 0.0)",
    )
    parser.add_argument(
        "--no-flip-v",
        action="store_true",
        help="Disable v-axis flip. By default v is flipped (v <- 1-v).",
    )

    parser.add_argument(
        "--out-npy",
        type=Path,
        default=Path("outputs/uv_position_map.npy"),
        help="Output path for raw position map .npy",
    )
    parser.add_argument(
        "--out-mask",
        type=Path,
        default=Path("outputs/uv_valid_mask.npy"),
        help="Output path for valid mask .npy",
    )
    parser.add_argument(
        "--out-pix2face",
        type=Path,
        default=Path("outputs/uv_pix2face.npy"),
        help="Output path for pix2face .npy",
    )
    parser.add_argument(
        "--out-bary",
        type=Path,
        default=Path("outputs/uv_barycentric.npy"),
        help="Output path for barycentric .npy",
    )
    parser.add_argument(
        "--out-vis",
        type=Path,
        default=Path("outputs/uv_position_map_vis.npy"),
        help=(
            "Output path for visualization array. "
            "Saved as uint8 NumPy file by default (you can convert to PNG later)."
        ),
    )

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    mesh = load_mesh_data(
        vertices_path=args.vertices,
        faces_path=args.faces,
        uv_vertices_path=args.uv_vertices,
        uv_faces_path=args.uv_faces,
    )

    uv_pos, valid_mask, pix2face, bary = generate_canonical_uv_position_map(
        mesh=mesh,
        resolution=args.resolution,
        flip_v=(not args.no_flip_v),
        background_value=args.background,
    )

    vis = normalize_for_visualization(uv_pos=uv_pos, valid_mask=valid_mask)

    ensure_parent(args.out_npy)
    ensure_parent(args.out_mask)
    ensure_parent(args.out_pix2face)
    ensure_parent(args.out_bary)
    ensure_parent(args.out_vis)

    np.save(args.out_npy, uv_pos)
    np.save(args.out_mask, valid_mask)
    np.save(args.out_pix2face, pix2face)
    np.save(args.out_bary, bary)
    np.save(args.out_vis, vis)

    print("[Done] Canonical UV position map generated.")
    print(f"  uv_position_map: {args.out_npy}")
    print(f"  uv_valid_mask:   {args.out_mask}")
    print(f"  uv_pix2face:     {args.out_pix2face}")
    print(f"  uv_barycentric:  {args.out_bary}")
    print(f"  uv_vis(uint8):   {args.out_vis}")


if __name__ == "__main__":
    main()
