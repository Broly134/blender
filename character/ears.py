"""Oreilles : coquille polaire posee sur la surface reelle du crane.

Le pavillon est genere comme un disque parametrique (rayon, angle) dont chaque
point est pousse vers l'exterieur depuis la surface exacte de la tete : c'est
ce qui evite l'effet "rondelle collee" quand on place une primitive a la main.
"""

import numpy as np

from . import anatomy as A
from .mathutil import bump, smoothstep
from .meshtools import object_from_arrays, grid_faces, add_solidify


N_R = 34
N_A = 80


def head_surface_x(y, z):
    """Demi-largeur du crane au point (y, z), d'apres le loft de base."""
    w = A.HALF_WIDTH(z)
    front = y >= 0.0
    depth = np.where(front, A.FRONT(z), A.BACK(z))
    n = np.where(front, A.EXP_FRONT(z), A.EXP_BACK(z))
    t = np.clip(np.abs(y) / np.maximum(depth, 1e-6), 0.0, 1.0)
    return w * np.clip(1.0 - t ** n, 0.0, 1.0) ** (1.0 / n)


def _outline(phi):
    """Silhouette du pavillon : ovale haut et large, lobe etroit en bas."""
    up, back = np.cos(phi), np.sin(phi)
    r = np.ones_like(phi)
    r += 0.06 * np.clip(up, 0.0, 1.0)
    r -= 0.26 * np.clip(-up, 0.0, 1.0) ** 1.4                       # lobe
    r -= 0.20 * np.clip(back, 0.0, 1.0) * np.clip(-up, 0.0, 1.0)
    r -= 0.24 * np.clip(-back, 0.0, 1.0) ** 1.2 * (0.35 + 0.65 * np.clip(-up, 0.0, 1.0))
    return r


def _relief(rr, phi):
    """Reliefs du pavillon, en metres, mesures depuis le plan du crane."""
    up, back = np.cos(phi), np.sin(phi)

    helix = bump((rr - 0.88) / 0.20) * (0.45 + 0.55 * smoothstep(-1.0, 0.5, up))
    anthelix = bump((rr - 0.58) / 0.26) * smoothstep(-0.35, 0.70, up)
    anthelix += 0.55 * bump((rr - 0.50) / 0.22) * smoothstep(0.0, 0.85, back)
    scapha = bump((rr - 0.74) / 0.14) * smoothstep(-0.2, 0.8, up)
    conque = bump(np.sqrt(np.clip(((rr - 0.20) / 0.36) ** 2
                                  + ((back + 0.35) / 1.15) ** 2, 0, 4)))
    tragus = bump(np.sqrt(np.clip(((rr - 0.42) / 0.26) ** 2
                                  + ((back + 0.95) / 0.45) ** 2
                                  + ((up + 0.05) / 0.70) ** 2, 0, 4)))
    lobe = bump(np.sqrt(np.clip((rr / 0.80) ** 2
                                + ((up + 0.92) / 0.45) ** 2, 0, 4)))

    return (0.0092 * helix + 0.0072 * anthelix - 0.0028 * scapha
            - 0.0048 * conque + 0.0042 * tragus + 0.0052 * lobe)


def build_ear(side=1.0):
    r = np.linspace(0.0, 1.0, N_R)
    phi = np.linspace(0.0, 2.0 * np.pi, N_A, endpoint=False)
    R, PHI = np.meshgrid(r, phi, indexing="ij")

    rr = R * _outline(PHI)

    half_h, half_w = 0.0318, 0.0158
    y = -rr * np.sin(PHI) * half_w
    z = rr * np.cos(PHI) * half_h

    # bascule arriere du pavillon
    ang = np.radians(16.0)
    c, s = np.cos(ang), np.sin(ang)
    y, z = y * c - z * s, y * s + z * c

    y = y + A.EAR_Y
    z = z + A.EAR_Z

    # on part de la surface exacte du crane, puis on ecarte le pavillon
    base = head_surface_x(y, z) - 0.0020
    lift = 0.0072 * smoothstep(0.30, 0.98, R)
    x = base + lift + _relief(R, PHI)

    verts = np.stack([x * side, y, z], axis=-1).reshape(-1, 3)
    faces = grid_faces(N_R, N_A, close_cols=True, flip=(side > 0))

    name = "OreilleG" if side > 0 else "OreilleD"
    return object_from_arrays(name, verts, faces)


def build_ears():
    objs = []
    for side in (1.0, -1.0):
        o = build_ear(side)
        add_solidify(o, 0.0024, offset=0.0)
        objs.append(o)
    return objs
