#!/usr/bin/env python3
"""Prepare Step-A input npy files from FastAvatar/FLAME assets.

This script bridges your dataset to the 4 minimal Step-A inputs:
- vertices.npy
- faces.npy
- uv_vertices.npy
- uv_faces.npy

It supports cases where the FLAME template OBJ has NO vt/uv data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


class ObjData:
    def __init__(
        self,
        vertices: np.ndarray,
        uv_vertices: np.ndarray,
        faces_v: np.ndarray,
        faces_vt: np.ndarray,
    ) -> None:
        self.vertices = vertices
        self.uv_vertices = uv_vertices
        self.faces_v = faces_v
        self.faces_vt = faces_vt



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Step-A inputs for canonical UV map")
    p.add_argument(
        "--flame-obj",
        type=Path,
        required=True,
        help="Geometry OBJ path (can be without vt), used for faces.npy",
    )
    p.add_argument(
        "--uv-obj",
        type=Path,
        default=None,
        help=(
            "Optional OBJ that contains UV data (vt + face vt index). "
            "Use this when --flame-obj has no vt."
        ),
    )

    p.add_argument(
        "--vertices-npy",
        type=Path,
        default=None,
        help="Optional direct vertices.npy path (preferred if you already have per-frame vertices)",
    )
    p.add_argument(
        "--frame-param",
        type=Path,
        default=None,
        help="Optional flame_param frame npz, e.g. flame_param/00000.npz",
    )
    p.add_argument(
        "--canonical-param",
        type=Path,
        default=None,
        help="Optional canonical_flame_param.npz",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/step_a_inputs"),
        help="Output folder for vertices/faces/uv_vertices/uv_faces npy",
    )
    return p.parse_args()


def _parse_face_token(tok: str) -> tuple[int, int | None]:
    parts = tok.split("/")
    v_idx = int(parts[0])

    vt_idx = None
    if len(parts) >= 2 and parts[1] != "":
        vt_idx = int(parts[1])

    return v_idx, vt_idx


def load_obj(obj_path: Path) -> ObjData:
    """Load OBJ with tolerant parsing for missing vt.

    Returns:
      vertices: (N,3)
      uv_vertices: (Nt,2) maybe empty
      faces_v: (M,3)
      faces_vt: (M,3), -1 if missing vt
    """
    vertices: list[list[float]] = []
    uv_vertices: list[list[float]] = []
    faces_v: list[list[int]] = []
    faces_vt: list[list[int]] = []

    with obj_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vt "):
                p = line.split()
                uv_vertices.append([float(p[1]), float(p[2])])
            elif line.startswith("f "):
                toks = line.split()[1:]
                if len(toks) != 3:
                    continue

                tri_v: list[int] = []
                tri_vt: list[int] = []
                has_any_vt = False

                for tok in toks:
                    v_idx, vt_idx = _parse_face_token(tok)
                    tri_v.append(v_idx - 1)
                    if vt_idx is None:
                        tri_vt.append(-1)
                    else:
                        tri_vt.append(vt_idx - 1)
                        has_any_vt = True

                faces_v.append(tri_v)
                # if no vt for this face, keep -1 placeholders
                if has_any_vt:
                    faces_vt.append(tri_vt)
                else:
                    faces_vt.append([-1, -1, -1])

    return ObjData(
        vertices=np.asarray(vertices, dtype=np.float32),
        uv_vertices=np.asarray(uv_vertices, dtype=np.float32),
        faces_v=np.asarray(faces_v, dtype=np.int32),
        faces_vt=np.asarray(faces_vt, dtype=np.int32),
    )


def _find_first_key(data: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> str | None:
    for k in candidates:
        if k in data.files:
            return k
    return None


def resolve_vertices_from_npz(
    frame_param_path: Path | None,
    canonical_param_path: Path | None,
) -> np.ndarray:
    """Try to resolve vertices from npz files.

    If vertices are not present, raise with clear instructions.
    """
    frame_keys: list[str] = []
    cano_keys: list[str] = []

    if frame_param_path is not None and frame_param_path.exists():
        frame = np.load(frame_param_path)
        frame_keys = list(frame.files)
        key = _find_first_key(frame, ["vertices", "verts", "v", "mesh_vertices"])
        if key is not None:
            verts = np.asarray(frame[key], dtype=np.float32)
            if verts.ndim == 2 and verts.shape[1] == 3:
                return verts

    if canonical_param_path is not None and canonical_param_path.exists():
        cano = np.load(canonical_param_path)
        cano_keys = list(cano.files)
        key2 = _find_first_key(cano, ["vertices", "verts", "v", "mesh_vertices"])
        if key2 is not None:
            verts = np.asarray(cano[key2], dtype=np.float32)
            if verts.ndim == 2 and verts.shape[1] == 3:
                return verts

    raise RuntimeError(
        "No vertices found in npz files.\n"
        "Provide --vertices-npy directly, OR add vertices key in npz, OR\n"
        "run FLAME forward using flame_param + flame2023.pkl to get (N,3) vertices.\n"
        f"frame_param keys: {frame_keys}\n"
        f"canonical_param keys: {cano_keys}"
    )


def _build_face_lookup_with_vt(faces_v: np.ndarray, faces_vt: np.ndarray) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    """Build lookup from geometry-face vertex triplet to uv-face vt triplet.

    We store all cyclic + reversed permutations so orientation differences can still match.
    """
    lut: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for fv, fvt in zip(faces_v, faces_vt):
        a, b, c = int(fv[0]), int(fv[1]), int(fv[2])
        ta, tb, tc = int(fvt[0]), int(fvt[1]), int(fvt[2])

        perms = [
            ((a, b, c), (ta, tb, tc)),
            ((b, c, a), (tb, tc, ta)),
            ((c, a, b), (tc, ta, tb)),
            ((a, c, b), (ta, tc, tb)),
            ((c, b, a), (tc, tb, ta)),
            ((b, a, c), (tb, ta, tc)),
        ]
        for k, v in perms:
            if k not in lut:
                lut[k] = v

    return lut


def _remap_uv_faces_by_geometry_faces(geom_faces_v: np.ndarray, uv_obj: ObjData) -> np.ndarray:
    """Remap UV faces to geometry faces when face counts differ.

    This works when both OBJs share the same vertex indexing for overlapping triangles,
    but uv_obj may contain extra triangles (e.g., extra mouth/teeth regions).
    """
    lut = _build_face_lookup_with_vt(uv_obj.faces_v, uv_obj.faces_vt)

    out = np.full((geom_faces_v.shape[0], 3), -1, dtype=np.int32)
    missing = 0

    for i, fv in enumerate(geom_faces_v):
        key = (int(fv[0]), int(fv[1]), int(fv[2]))
        mapped = lut.get(key)
        if mapped is None:
            missing += 1
            continue
        out[i] = np.asarray(mapped, dtype=np.int32)

    if missing > 0:
        raise RuntimeError(
            "Failed to remap some geometry faces to UV faces. "
            f"missing={missing}/{geom_faces_v.shape[0]}. "
            "Likely the two OBJs are not topology-compatible."
        )

    return out


def pick_uv_topology(geom_obj: ObjData, uv_obj: ObjData | None) -> tuple[np.ndarray, np.ndarray]:
    """Choose uv_vertices + uv_faces.

    Priority:
    1) use uv data from geom_obj if complete
    2) else use uv_obj (if provided)
       - if face counts equal: direct use
       - else: try remap by geometry face indices
    """
    geom_has_uv = geom_obj.uv_vertices.size > 0 and np.all(geom_obj.faces_vt >= 0)
    if geom_has_uv:
        return geom_obj.uv_vertices, geom_obj.faces_vt

    if uv_obj is None:
        raise RuntimeError(
            "Geometry OBJ has no usable UV (vt) data.\n"
            "Please provide --uv-obj path (for example your frame flame.obj that contains vt)."
        )

    uv_has_uv = uv_obj.uv_vertices.size > 0 and np.all(uv_obj.faces_vt >= 0)
    if not uv_has_uv:
        raise RuntimeError("--uv-obj also has no usable UV (vt) data.")

    if uv_obj.faces_v.shape[0] == geom_obj.faces_v.shape[0]:
        return uv_obj.uv_vertices, uv_obj.faces_vt

    # Face count mismatch: attempt robust remap by geometry face vertex ids.
    remapped_uv_faces = _remap_uv_faces_by_geometry_faces(geom_obj.faces_v, uv_obj)
    return uv_obj.uv_vertices, remapped_uv_faces


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    geom_obj = load_obj(args.flame_obj)
    uv_obj = load_obj(args.uv_obj) if args.uv_obj is not None else None

    faces = geom_obj.faces_v
    uv_vertices, uv_faces = pick_uv_topology(geom_obj, uv_obj)

    np.save(out / "faces.npy", faces)
    np.save(out / "uv_vertices.npy", uv_vertices)
    np.save(out / "uv_faces.npy", uv_faces)

    vertices = None
    if args.vertices_npy is not None:
        vertices = np.asarray(np.load(args.vertices_npy), dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(
                f"--vertices-npy must be (N,3), got {vertices.shape}"
            )
        print("[OK] vertices loaded from --vertices-npy")
    else:
        try:
            vertices = resolve_vertices_from_npz(args.frame_param, args.canonical_param)
            print("[OK] vertices resolved from npz")
        except RuntimeError as e:
            print("[WARN] vertices.npy not generated.")
            print(str(e))

    if vertices is not None:
        np.save(out / "vertices.npy", vertices)

    print("[Done] Generated files:")
    print(f"  faces.npy:       {out / 'faces.npy'}")
    print(f"  uv_vertices.npy: {out / 'uv_vertices.npy'}")
    print(f"  uv_faces.npy:    {out / 'uv_faces.npy'}")
    if vertices is not None:
        print(f"  vertices.npy:    {out / 'vertices.npy'}")
    else:
        print("  vertices.npy:    [missing - please provide via --vertices-npy or FLAME forward]")


if __name__ == "__main__":
    main()
