"""Buste habille : epaules, t-shirt gris et veste a capuche noire ouverte.

Aucune decoupe de faces ici : l'encolure du t-shirt et l'ouverture en V de la
veste sont obtenues en faisant dependre la hauteur du bord superieur de l'angle
azimutal. Le bord reste donc parfaitement net, la ou une suppression de faces
laisserait un escalier.
"""

import numpy as np

from .mathutil import Profile, smoothstep
from .meshtools import object_from_arrays, grid_faces, cap_ring, add_solidify, join_objects


BOTTOM = -0.5200
N_TH = 140
N_Z = 60

HALF_W = Profile([
    (-0.5200, 0.2280),
    (-0.4200, 0.2320),
    (-0.3400, 0.2330),
    (-0.2700, 0.2270),
    (-0.2200, 0.2040),
    (-0.1800, 0.1610),
    (-0.1500, 0.1140),
    (-0.1250, 0.0824),
    (-0.1000, 0.0688),
    (-0.0700, 0.0646),
    (-0.0480, 0.0632),
])

FRONT = Profile([
    (-0.5200, 0.1100),
    (-0.4200, 0.1160),
    (-0.3400, 0.1150),
    (-0.2700, 0.1080),
    (-0.2200, 0.0975),
    (-0.1800, 0.0800),
    (-0.1500, 0.0552),
    (-0.1250, 0.0342),
    (-0.1000, 0.0248),
    (-0.0700, 0.0224),
    (-0.0480, 0.0234),
])

BACK = Profile([
    (-0.5200, 0.1180),
    (-0.4200, 0.1230),
    (-0.3400, 0.1250),
    (-0.2700, 0.1230),
    (-0.2200, 0.1165),
    (-0.1800, 0.1045),
    (-0.1500, 0.0878),
    (-0.1250, 0.0758),
    (-0.1000, 0.0710),
    (-0.0700, 0.0704),
    (-0.0480, 0.0714),
])


def _front_amount(theta):
    return np.clip(np.sin(theta), 0.0, 1.0)


def _torso_arrays(top_fn, scale=1.0, thick=0.0, folds=0.0):
    """Loft dont le bord superieur suit une courbe donnee (encolure, col en V)."""
    th = np.linspace(0.0, 2.0 * np.pi, N_TH, endpoint=False)
    t = np.linspace(0.0, 1.0, N_Z)
    T, TH = np.meshgrid(t, th, indexing="ij")

    top = top_fn(TH)
    Z = BOTTOM + (top - BOTTOM) * T

    w = HALF_W(Z) * scale + thick
    f = FRONT(Z) * scale + thick
    b = BACK(Z) * scale + thick

    c, s = np.cos(TH), np.sin(TH)
    e = 2.0 / 2.45
    x = w * np.sign(c) * np.abs(c) ** e
    y = np.where(s >= 0, f, b) * np.sign(s) * np.abs(s) ** e

    if folds:
        # plis de tissu : ondulation basse frequence sur la surface
        ripple = (np.sin(7.0 * TH + 2.1) * np.sin(11.0 * Z * 6.0)
                  + 0.6 * np.sin(13.0 * TH) * np.sin(23.0 * Z * 6.0))
        r = 1.0 + folds * ripple
        x, y = x * r, y * r

    # les epaules tombent
    drop = smoothstep(-0.24, -0.15, Z) * np.clip(np.abs(c), 0.0, 1.0) ** 1.4
    Z = Z - 0.028 * drop
    return np.stack([x, y, Z], axis=-1).reshape(-1, 3)


def _collar(top_fn, height, thick, folds=0.05):
    """Bande de col : elle repart du bord superieur du vetement et remonte,
    en suivant exactement le meme profil. Aucun ecart possible avec le cou."""
    th = np.linspace(0.0, 2.0 * np.pi, N_TH, endpoint=False)
    v = np.linspace(0.0, 1.0, 6)
    V, TH = np.meshgrid(v, th, indexing="ij")

    z = top_fn(TH) + height * V
    w = HALF_W(z) + thick
    f = FRONT(z) + thick
    b = BACK(z) + thick

    c, s = np.cos(TH), np.sin(TH)
    e = 2.0 / 2.45
    r = 1.0 + folds * 0.06 * np.sin(11.0 * TH)
    x = w * r * np.sign(c) * np.abs(c) ** e
    y = np.where(s >= 0, f, b) * r * np.sign(s) * np.abs(s) ** e
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return verts, grid_faces(6, N_TH, close_cols=True)


def _shirt_top(theta):
    return -0.0760 - 0.0230 * _front_amount(theta) ** 1.8


def _jacket_top(theta):
    return -0.1180 - 0.1450 * _front_amount(theta) ** 2.1


def build_clothing():
    from .materials import skin_material, fabric_material, metal_material
    from .skin_attributes import paint

    parts = []

    # --- peau du buste, visible dans l'encolure ---------------------------
    v = _torso_arrays(lambda th: np.full_like(th, -0.0480), scale=0.985)
    faces = grid_faces(N_Z, N_TH, close_cols=True)
    faces.append(cap_ring(0, N_TH, flip=True))
    torso = object_from_arrays("Buste", v, faces)
    paint(torso)
    torso.data.materials.append(skin_material())
    parts.append(torso)

    tee_mat = fabric_material("TissuTShirt", color=(0.0225, 0.0212, 0.0198),
                              sheen=0.16, scale=520.0, roughness=0.88)
    jacket_mat = fabric_material("TissuVeste", color=(0.0098, 0.0099, 0.0108),
                                 sheen=0.22, scale=380.0, roughness=0.82)

    # --- t-shirt ----------------------------------------------------------
    v = _torso_arrays(_shirt_top, thick=0.0048, folds=0.010)
    shirt = object_from_arrays("TShirt", v, grid_faces(N_Z, N_TH, close_cols=True))
    add_solidify(shirt, 0.0022, offset=0.0)
    shirt.data.materials.append(tee_mat)
    parts.append(shirt)

    v, f = _collar(_shirt_top, 0.0165, 0.0072)
    collar = object_from_arrays("ColTShirt", v, f)
    add_solidify(collar, 0.0018, offset=0.0)
    collar.data.materials.append(tee_mat)
    parts.append(collar)

    # --- veste ------------------------------------------------------------
    v = _torso_arrays(_jacket_top, thick=0.0185, folds=0.017)
    jacket = object_from_arrays("Veste", v, grid_faces(N_Z, N_TH, close_cols=True))
    add_solidify(jacket, 0.0034, offset=0.0)
    jacket.data.materials.append(jacket_mat)
    parts.append(jacket)

    # --- capuche affaissee derriere la nuque ------------------------------
    hood = _hood_roll()
    hood.data.materials.append(jacket_mat)
    parts.append(hood)

    # --- fermeture eclair -------------------------------------------------
    zipper = _zipper()
    zipper.data.materials.append(metal_material("MetalZip", color=(0.17, 0.17, 0.18),
                                                roughness=0.52))
    parts.append(zipper)
    return parts


def _hood_roll():
    n_u, n_v = 64, 26
    u = np.linspace(-1.0, 1.0, n_u)
    v = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")

    ang = U * np.radians(118.0) + np.pi * 1.5
    cx = 0.112 * np.cos(ang)
    cy = -0.032 + 0.100 * np.sin(ang)
    cz = -0.126 - 0.062 * U ** 2

    r = 0.042 * (1.0 - 0.28 * U ** 2)
    r = r * (1.0 + 0.11 * np.sin(9.0 * U + 1.3) + 0.06 * np.sin(5.0 * V))

    nx, ny = np.cos(ang), np.sin(ang)
    x = cx + nx * r * np.cos(V)
    y = cy + ny * r * np.cos(V)
    z = cz + r * 0.88 * np.sin(V)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return object_from_arrays("Capuche", verts, grid_faces(n_u, n_v, close_cols=True))


def _zipper():
    parts = []
    z = np.linspace(-0.130, -0.380, 56)
    # l'ouverture s'evase vers le bas, comme le bord de la veste
    spread = 0.012 + 0.145 * smoothstep(-0.130, -0.310, z)
    for sx in (1.0, -1.0):
        cx = sx * spread
        cy = FRONT(z) + 0.0205
        n_s = 8
        a = np.linspace(0.0, 2.0 * np.pi, n_s, endpoint=False)
        verts = np.empty((len(z), n_s, 3))
        for i in range(len(z)):
            verts[i, :, 0] = cx[i] + 0.0019 * np.cos(a)
            verts[i, :, 1] = cy[i] + 0.0015 * np.sin(a)
            verts[i, :, 2] = z[i]
        parts.append(object_from_arrays(f"Glissiere{sx:+.0f}",
                                        verts.reshape(-1, 3),
                                        grid_faces(len(z), n_s, close_cols=True)))
    return join_objects(parts, "Fermeture")
