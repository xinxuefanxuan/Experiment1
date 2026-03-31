#!/usr/bin/env python3
"""One-command pipeline: single image -> Step-A UV position map.

This script orchestrates:
1) (Optional) run an external FLAME estimator from one image to produce vertices.npy
2) prepare Step-A inputs (faces/uv_vertices/uv_faces/vertices)
3) generate canonical UV position map outputs

It is designed to be framework-agnostic: you can plug DECA/EMOCA/VHAP/your tracker
through --estimator-cmd.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single image -> canonical UV position map")
    p.add_argument("--image", type=Path, required=True, help="Input image path")
    p.add_argument("--flame-obj", type=Path, required=True, help="Geometry OBJ path")
    p.add_argument("--uv-obj", type=Path, default=None, help="Optional UV OBJ path")

    p.add_argument(
        "--vertices-npy",
        type=Path,
        default=None,
        help="Direct vertices.npy path (if already available)",
    )
    p.add_argument(
        "--estimator-cmd",
        type=str,
        default=None,
        help=(
            "Optional external command to estimate vertices from image. "
            "Template vars: {image}, {work_dir}, {vertices_out}."
        ),
    )
    p.add_argument(
        "--estimator-vertices-out",
        type=Path,
        default=Path("work/vertices.npy"),
        help="Expected vertices output path from estimator command",
    )

    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--step-a-dir", type=Path, default=Path("data/step_a_input"))
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def run_cmd(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {cmd}")


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.step_a_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    vertices_path = args.vertices_npy

    if vertices_path is None:
        if not args.estimator_cmd:
            raise RuntimeError(
                "Need either --vertices-npy OR --estimator-cmd to get vertices from image."
            )

        est_vertices = args.estimator_vertices_out
        est_vertices.parent.mkdir(parents=True, exist_ok=True)

        cmd = args.estimator_cmd.format(
            image=str(args.image),
            work_dir=str(args.work_dir),
            vertices_out=str(est_vertices),
        )
        run_cmd(cmd)

        if not est_vertices.exists():
            raise RuntimeError(
                f"Estimator finished but vertices file not found: {est_vertices}"
            )
        vertices_path = est_vertices

    prep_parts = [
        "PYTHONPATH=src python tools/prepare_step_a_inputs_from_fastavatar.py",
        f"--flame-obj {shlex.quote(str(args.flame_obj))}",
        f"--vertices-npy {shlex.quote(str(vertices_path))}",
        f"--out-dir {shlex.quote(str(args.step_a_dir))}",
    ]
    if args.uv_obj is not None:
        prep_parts.append(f"--uv-obj {shlex.quote(str(args.uv_obj))}")
    run_cmd(" ".join(prep_parts))

    gen_cmd = " ".join(
        [
            "PYTHONPATH=src python tools/generate_canonical_uv_position_map.py",
            f"--vertices {shlex.quote(str(args.step_a_dir / 'vertices.npy'))}",
            f"--faces {shlex.quote(str(args.step_a_dir / 'faces.npy'))}",
            f"--uv-vertices {shlex.quote(str(args.step_a_dir / 'uv_vertices.npy'))}",
            f"--uv-faces {shlex.quote(str(args.step_a_dir / 'uv_faces.npy'))}",
            f"--resolution {args.resolution}",
            f"--background {args.background}",
            f"--out-npy {shlex.quote(str(args.outputs_dir / 'uv_position_map.npy'))}",
            f"--out-mask {shlex.quote(str(args.outputs_dir / 'uv_valid_mask.npy'))}",
            f"--out-pix2face {shlex.quote(str(args.outputs_dir / 'uv_pix2face.npy'))}",
            f"--out-bary {shlex.quote(str(args.outputs_dir / 'uv_barycentric.npy'))}",
            f"--out-vis {shlex.quote(str(args.outputs_dir / 'uv_position_map_vis.npy'))}",
        ]
    )
    run_cmd(gen_cmd)

    print("[DONE] single-image pipeline finished.")
    print(f"  step_a inputs: {args.step_a_dir}")
    print(f"  outputs:       {args.outputs_dir}")


if __name__ == "__main__":
    main()
