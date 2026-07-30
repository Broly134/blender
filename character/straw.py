"""Brin de ble tenu au coin de la bouche : chaume, epi et barbes."""

import numpy as np

from . import anatomy as A
from .mathutil import smoothstep
from .meshtools import object_from_arrays, grid_faces, join_objects


def _tube(path, radii, n_s=10):
    n = len(path)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    normal = np.cross(tangent, ref)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    binormal = np.cross(normal, tangent)

    a = np.linspace(0.0, 2.0 * np.pi, n_s, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    verts = np.empty((n, n_s, 3))
    for i in range(n):
        verts[i] = (path[i][None, :]
                    + normal[i][None, :] * (ca * radii[i])[:, None]
                    + binormal[i][None, :] * (sa * radii[i])[:, None])
    return verts.reshape(-1, 3), grid_faces(n, n_s, close_cols=True)


def _stalk_path(n=70):
    """Part de la commissure gauche du personnage, part vers l'exterieur et le bas."""
    key = np.array([
        [-0.0135, 0.0745, A.STOMION - 0.0010],
        [-0.0215, 0.0835, A.STOMION - 0.0030],
        [-0.0310, 0.0900, A.STOMION - 0.0105],
        [-0.0415, 0.0935, A.STOMION - 0.0225],
        [-0.0540, 0.0945, A.STOMION - 0.0380],
        [-0.0672, 0.0930, A.STOMION - 0.0560],
        [-0.0800, 0.0895, A.STOMION - 0.0762],
    ])
    tk = np.linspace(0.0, 1.0, len(key))
    t = np.linspace(0.0, 1.0, n)
    path = np.stack([np.interp(t, tk, key[:, i]) for i in range(3)], axis=1)
    for _ in range(30):
        path[1:-1] = 0.5 * path[1:-1] + 0.25 * (path[:-2] + path[2:])
    return path, t


def _grain(center, direction, size):
    """Un grain de ble : ellipsoide effile, oriente le long de l'epi."""
    rings, seg = 12, 12
    phi = np.linspace(0.0, np.pi, rings)
    th = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    P, T = np.meshgrid(phi, th, indexing="ij")
    x = np.sin(P) * np.cos(T) * size[0]
    y = np.sin(P) * np.sin(T) * size[1]
    z = np.cos(P) * size[2]
    # le grain est plus pointu vers l'exterieur
    z = z - size[2] * 0.35 * (1.0 - np.cos(P))
    v = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    if abs(d @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, d)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    basis = np.stack([e1, e2, d], axis=1)
    return v @ basis.T + np.asarray(center), grid_faces(rings, seg, close_cols=True)


def build_straw():
    from .materials import straw_material

    path, t = _stalk_path()
    radii = 0.00062 * (1.0 - 0.35 * t) + 0.00010
    v, f = _tube(path, radii)
    parts = [object_from_arrays("Chaume", v, f)]

    # --- epi : deux rangees de grains alternes le long de la fin de la tige --
    tail = path[int(0.52 * len(path)):]
    tan = np.gradient(tail, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    side = np.cross(tan, np.tile(np.array([0.0, 1.0, 0.0]), (len(tail), 1)))
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)

    rng = np.random.default_rng(4)
    n_grain = 15
    idx = np.linspace(0, len(tail) - 3, n_grain).astype(int)
    awns = []
    for k, i in enumerate(idx):
        sgn = 1.0 if k % 2 == 0 else -1.0
        grow = smoothstep(0.0, 0.35, k / n_grain) * smoothstep(1.02, 0.72, k / n_grain)
        scale = 0.55 + 0.45 * grow
        base = tail[i] + side[i] * sgn * 0.0016
        direction = (tan[i] * 0.55 + side[i] * sgn * 0.80
                     + np.array([0.0, 0.0, 0.10]))
        v, f = _grain(base + direction * 0.0022,
                      direction,
                      (0.00092 * scale, 0.00078 * scale, 0.0026 * scale))
        parts.append(object_from_arrays(f"Grain{k}", v, f))

        # barbe (l'arete fine qui prolonge chaque grain)
        tip = base + direction / np.linalg.norm(direction) * 0.0058 * scale
        jitter = rng.normal(scale=0.06, size=3)
        end = tip + (direction / np.linalg.norm(direction) + jitter) * 0.0155 * scale
        seg = np.linspace(0.0, 1.0, 12)[:, None]
        awn_path = tip[None, :] * (1 - seg) + end[None, :] * seg
        awn_path[:, 2] -= 0.004 * seg[:, 0] ** 2
        vr = 0.00016 * (1.0 - 0.85 * seg[:, 0]) + 0.00003
        v, f = _tube(awn_path, vr, n_s=5)
        awns.append(object_from_arrays(f"Barbe{k}", v, f))

    straw = join_objects(parts + awns, "BrinDeBle")
    straw.data.materials.append(straw_material())
    return [straw]
