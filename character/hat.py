"""Chapeau de cowboy en feutre : calotte a pli "cattleman", bord releve, bandeau.

La calotte est un loft super-elliptique (comme le crane), puis on y enfonce le
pli central et les deux pincements lateraux, exactement comme on marque un
feutre a la main. Le bord est une nappe dont la hauteur du liseré depend de
l'angle : plongeante devant, franchement relevee sur les cotes.
"""

import numpy as np

from .mathutil import smoothstep
from .meshtools import object_from_arrays, grid_faces, cap_ring, add_solidify


Z_BASE = 0.1545         # bas de la calotte, la ou elle coiffe le crane
CROWN_H = 0.1075
W0, D0 = 0.0935, 0.1055  # demi-largeur / demi-profondeur de la calotte
Y0 = -0.0075             # la calotte est legerement reculee
TILT = np.radians(8.5)  # chapeau porte rejete en arriere

N_TH = 144


def _crown_arrays():
    n_wall, n_cap = 40, 16
    theta = np.linspace(0.0, 2.0 * np.pi, N_TH, endpoint=False)

    # --- paroi ------------------------------------------------------------
    t = np.linspace(0.0, 1.0, n_wall)
    z_wall = Z_BASE + CROWN_H * 0.90 * t
    # le feutre s'evase tres legerement vers le bas puis se resserre en haut
    s_wall = 1.0 + 0.012 * np.exp(-8.0 * t) - 0.115 * t ** 1.35

    # --- calotte ----------------------------------------------------------
    u = np.linspace(0.0, 1.0, n_cap + 1)[1:]
    z_cap = Z_BASE + CROWN_H * (0.90 + 0.10 * np.sin(u * np.pi / 2.0))
    s_cap = s_wall[-1] * np.cos(u * np.pi / 2.0) ** 0.42

    z = np.concatenate([z_wall, z_cap])
    s = np.concatenate([s_wall, s_cap])

    Z, T = np.meshgrid(z, theta, indexing="ij")
    S = np.repeat(s[:, None], N_TH, axis=1)

    c, sn = np.cos(T), np.sin(T)
    e = 2.0 / 2.45                      # section presque ovale, un peu carree
    x = W0 * S * np.sign(c) * np.abs(c) ** e
    y = Y0 + D0 * S * np.sign(sn) * np.abs(sn) ** e
    return np.stack([x, y, Z], axis=-1).reshape(-1, 3), len(z)


def _crease(v):
    """Pli cattleman : gouttiere centrale + deux pincements avant."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    top = Z_BASE + CROWN_H

    # profondeur du pli : nulle en bas, maximale sur la calotte
    d = smoothstep(Z_BASE + CROWN_H * 0.52, top - 0.004, z)

    # gouttiere longitudinale
    groove = np.exp(-0.5 * (x / 0.0255) ** 2)
    groove *= smoothstep(0.115, 0.075, np.abs(y - Y0))
    v[:, 2] -= 0.0108 * groove * d

    # bourrelets de part et d'autre de la gouttiere
    ridge = np.exp(-0.5 * ((np.abs(x) - 0.045) / 0.020) ** 2)
    v[:, 2] += 0.0038 * ridge * d * smoothstep(0.115, 0.080, np.abs(y - Y0))

    # pincements lateraux avant (les deux "dents" du cattleman)
    for sx in (1.0, -1.0):
        pinch = np.exp(-0.5 * (((x - sx * 0.072) / 0.030) ** 2
                               + ((y - (Y0 + 0.052)) / 0.038) ** 2))
        pinch *= smoothstep(Z_BASE + CROWN_H * 0.60, top, z)
        v[:, 0] -= sx * 0.0052 * pinch
        v[:, 2] -= 0.0026 * pinch

    # avant de la calotte legerement affaisse
    front = np.exp(-0.5 * ((y - (Y0 + 0.085)) / 0.045) ** 2)
    v[:, 2] -= 0.0058 * front * smoothstep(Z_BASE + CROWN_H * 0.70, top, z)
    return v


def _brim_arrays():
    n_t = 26
    theta = np.linspace(0.0, 2.0 * np.pi, N_TH, endpoint=False)
    t = np.linspace(0.0, 1.0, n_t)
    T, TH = np.meshgrid(t, theta, indexing="ij")

    c, sn = np.cos(TH), np.sin(TH)
    e = 2.0 / 2.45
    ux = np.sign(c) * np.abs(c) ** e
    uy = np.sign(sn) * np.abs(sn) ** e

    # du bas de la calotte jusqu'au liseré
    r_in_x, r_in_y = W0 * 1.005, D0 * 1.005
    r_out_x, r_out_y = 0.1490, 0.1570

    x = (r_in_x + (r_out_x - r_in_x) * T) * ux
    y = Y0 + (r_in_y + (r_out_y - r_in_y) * T) * uy

    front = np.clip(sn, 0.0, 1.0)
    back = np.clip(-sn, 0.0, 1.0)
    side = np.abs(c)

    # hauteur du liseré : plongeante devant, tres relevee sur les cotes
    edge = (-0.0105 * front ** 1.7 + 0.0705 * side ** 1.28 + 0.0245 * back ** 1.4)
    # petit retroussis juste au bord
    z = Z_BASE - 0.0075 + edge * T ** 2.1 + 0.0060 * side * T ** 6.0
    # legere ondulation, un feutre n'est jamais parfaitement regulier
    z += 0.0035 * np.sin(3.0 * TH + 0.7) * T ** 2.4

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return verts, n_t


def _band_arrays():
    """Bandeau de cuir a la base de la calotte."""
    n_v = 8
    theta = np.linspace(0.0, 2.0 * np.pi, N_TH, endpoint=False)
    vv = np.linspace(0.0, 1.0, n_v)
    V, TH = np.meshgrid(vv, theta, indexing="ij")

    c, sn = np.cos(TH), np.sin(TH)
    e = 2.0 / 2.45
    z = Z_BASE + 0.0060 + 0.0210 * V
    s = 1.0 + 0.012 * np.exp(-8.0 * (V * 0.17)) - 0.115 * (V * 0.17) ** 1.35
    s = s * 1.012
    x = W0 * s * np.sign(c) * np.abs(c) ** e
    y = Y0 + D0 * s * np.sign(sn) * np.abs(sn) ** e
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return verts, n_v


def _rivets():
    from .meshtools import uv_sphere_arrays
    parts = []
    for k, ang in enumerate(np.radians([62.0, 78.0, 102.0, 118.0])):
        c, sn = np.cos(ang), np.sin(ang)
        e = 2.0 / 2.45
        s = 1.020
        cx = W0 * s * np.sign(c) * np.abs(c) ** e
        cy = Y0 + D0 * s * np.sign(sn) * np.abs(sn) ** e
        v, f = uv_sphere_arrays(radius=0.0026, rings=10, segments=14)
        v[:, 2] *= 0.45
        v = v + np.array([cx, cy, Z_BASE + 0.0165])
        parts.append(object_from_arrays(f"Rivet{k}", v, f))
    return parts


def build_hat():
    from .materials import felt_material, leather_material, metal_material
    from .meshtools import join_objects

    crown_v, n_rows = _crown_arrays()
    _crease(crown_v)
    faces = grid_faces(n_rows, N_TH, close_cols=True)
    faces.append(cap_ring(len(crown_v) - N_TH, N_TH))
    crown = object_from_arrays("Calotte", crown_v, faces)

    brim_v, n_t = _brim_arrays()
    brim = object_from_arrays("Bord", brim_v, grid_faces(n_t, N_TH, close_cols=True))
    add_solidify(brim, 0.0042, offset=0.0)

    band_v, n_v = _band_arrays()
    band = object_from_arrays("Bandeau", band_v, grid_faces(n_v, N_TH, close_cols=True))

    felt = felt_material()
    for o in (crown, brim):
        o.data.materials.append(felt)
    band.data.materials.append(leather_material())

    rivets = _rivets()
    metal = metal_material("LaitonRivet", color=(0.72, 0.60, 0.36), roughness=0.30)
    for r in rivets:
        r.data.materials.append(metal)
    rivet = join_objects(rivets, "Rivets")

    hat = join_objects([crown, brim], "Chapeau")

    for o in (hat, band, rivet):
        # rotation positive : le devant du bord se releve
        o.rotation_euler = (TILT, 0.0, np.radians(2.0))
        o.location = (0.0, 0.0105, 0.0010)
    return [hat, band, rivet]
