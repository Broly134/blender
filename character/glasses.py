"""Lunettes a monture epaisse noire (type acetate rectangulaire).

Cercles, pont et branches sont tous obtenus en balayant un profil rectangulaire
arrondi le long d'un chemin : c'est la meme fonction pour les trois.
"""

import numpy as np

from . import anatomy as A
from .meshtools import object_from_arrays, grid_faces, join_objects


LENS_X = 0.0358          # centre du verre
LENS_Z = A.EYE_LINE + 0.0012
LENS_Y = 0.0880          # plan avant de la monture
HALF_W = 0.0302
HALF_H = 0.0228
CORNER = 0.0072

RIM_T = 0.0082           # epaisseur radiale de la monture
RIM_D = 0.0072           # profondeur (selon Y)


def _rounded_rect(n, half_w, half_h, corner):
    """Contour ferme d'un rectangle arrondi, echantillonne regulierement."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / 4.4
    u = np.sign(c) * np.abs(c) ** e
    w = np.sign(s) * np.abs(s) ** e
    return u * half_w, w * half_h


def _sweep(path, normal, binormal, prof_u, prof_v, close_path=True):
    """Balaye un profil 2D le long d'un chemin 3D."""
    n_p, n_s = len(path), len(prof_u)
    verts = np.empty((n_p, n_s, 3))
    for i in range(n_p):
        verts[i] = (path[i][None, :]
                    + normal[i][None, :] * prof_u[:, None]
                    + binormal[i][None, :] * prof_v[:, None])
    faces = grid_faces(n_p, n_s, close_cols=True)
    if close_path:
        # referme le tube sur lui-meme
        for j in range(n_s):
            j2 = (j + 1) % n_s
            faces.append(((n_p - 1) * n_s + j, (n_p - 1) * n_s + j2, j2, j))
    return verts.reshape(-1, 3), faces


def _tube_profile(n=14, half_u=RIM_T * 0.5, half_v=RIM_D * 0.5):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / 4.5
    return np.sign(c) * np.abs(c) ** e * half_u, np.sign(s) * np.abs(s) ** e * half_v


def _lens_ring(side):
    n = 96
    ux, uz = _rounded_rect(n, HALF_W, HALF_H, CORNER)
    cx = side * LENS_X + ux
    cz = LENS_Z + uz
    # la face avant de la monture suit un cylindre tres ouvert
    cy = LENS_Y - 0.055 * (1.0 - np.cos(np.arcsin(np.clip(cx / 0.13, -1, 1))))

    path = np.stack([cx, cy, cz], axis=1)
    radial = np.stack([ux, np.zeros(n), uz], axis=1)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    binormal = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    pu, pv = _tube_profile()
    return _sweep(path, radial, binormal, pu, pv)


def _lens_glass(side):
    n_r, n_a = 12, 96
    ux, uz = _rounded_rect(n_a, HALF_W - RIM_T * 0.45, HALF_H - RIM_T * 0.45, CORNER)
    r = np.linspace(0.0, 1.0, n_r)
    X = side * LENS_X + r[:, None] * ux[None, :]
    Z = LENS_Z + r[:, None] * uz[None, :]
    base = LENS_Y - 0.055 * (1.0 - np.cos(np.arcsin(np.clip(X / 0.13, -1, 1))))
    Y = base + 0.0014 * (1.0 - r[:, None] ** 2) - 0.0006
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return verts, grid_faces(n_r, n_a, close_cols=True)


def _bridge():
    n = 26
    t = np.linspace(0.0, 1.0, n)
    x = np.linspace(-LENS_X + HALF_W - 0.0015, LENS_X - HALF_W + 0.0015, n)
    z = LENS_Z + HALF_H - 0.0060 + 0.0075 * np.sin(t * np.pi) ** 1.4
    y = LENS_Y - 0.0035 - 0.0060 * np.sin(t * np.pi)
    path = np.stack([x, y, z], axis=1)
    up = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    fwd = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    pu, pv = _tube_profile(half_u=0.0034, half_v=0.0036)
    return _sweep(path, up, fwd, pu, pv, close_path=False)


def _temple(side):
    """Branche : charniere, tige droite, puis crochet derriere l'oreille."""
    key = np.array([
        [side * (LENS_X + HALF_W - 0.0010), LENS_Y - 0.0060, LENS_Z + HALF_H - 0.0075],
        [side * (LENS_X + HALF_W + 0.0060), LENS_Y - 0.0180, LENS_Z + HALF_H - 0.0090],
        [side * 0.0710, 0.0530, LENS_Z + 0.0060],
        [side * 0.0760, 0.0060, LENS_Z + 0.0020],
        [side * 0.0755, -0.0230, LENS_Z - 0.0035],
        [side * 0.0720, -0.0330, LENS_Z - 0.0140],
        [side * 0.0680, -0.0300, LENS_Z - 0.0245],
    ])
    n = 64
    tk = np.linspace(0.0, 1.0, len(key))
    t = np.linspace(0.0, 1.0, n)
    path = np.stack([np.interp(t, tk, key[:, i]) for i in range(3)], axis=1)
    # lissage du chemin
    for _ in range(24):
        path[1:-1] = 0.5 * path[1:-1] + 0.25 * (path[:-2] + path[2:])

    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    up = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    normal = np.cross(tangent, up)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    binormal = np.cross(normal, tangent)

    taper = 1.0 - 0.35 * t
    pu, pv = _tube_profile(half_u=0.0018, half_v=0.0034)
    n_s = len(pu)
    verts = np.empty((n, n_s, 3))
    for i in range(n):
        verts[i] = (path[i][None, :]
                    + normal[i][None, :] * (pu * taper[i])[:, None]
                    + binormal[i][None, :] * (pv * taper[i])[:, None])
    faces = grid_faces(n, n_s, close_cols=True)
    return verts.reshape(-1, 3), faces


def build_glasses():
    from .materials import plastic_material, lens_material

    frame_parts = []
    for side in (1.0, -1.0):
        v, f = _lens_ring(side)
        frame_parts.append(object_from_arrays(f"Cercle{side:+.0f}", v, f))
        v, f = _temple(side)
        frame_parts.append(object_from_arrays(f"Branche{side:+.0f}", v, f))
    v, f = _bridge()
    frame_parts.append(object_from_arrays("Pont", v, f))

    frame = join_objects(frame_parts, "MontureLunettes")
    frame.data.materials.append(plastic_material())

    glass_parts = []
    for side in (1.0, -1.0):
        v, f = _lens_glass(side)
        glass_parts.append(object_from_arrays(f"Verre{side:+.0f}", v, f))
    glass = join_objects(glass_parts, "Verres")
    glass.data.materials.append(lens_material())

    for o in (frame, glass):
        o.rotation_euler = (np.radians(-6.0), 0.0, 0.0)
        o.location = (0.0, 0.0018, 0.0030)
    return [frame, glass]
