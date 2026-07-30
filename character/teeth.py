"""Arcade dentaire superieure, gencive et fond de bouche.

Sur la photo la bouche est entrouverte : on voit le bord des incisives et une
zone sombre en dessous. C'est exactement ce qu'il faut reproduire, rien de plus.
"""

import numpy as np

from . import anatomy as A
from .meshtools import object_from_arrays, grid_faces, join_objects


# demi-arcade : (demi-largeur de la dent, hauteur de couronne, avancee)
UPPER_TEETH = [
    (0.00435, 0.0102, 0.0000),   # incisive centrale
    (0.00335, 0.0086, -0.0004),  # incisive laterale
    (0.00390, 0.0098, -0.0002),  # canine
    (0.00360, 0.0082, -0.0008),  # premolaire
    (0.00370, 0.0074, -0.0012),
    (0.00450, 0.0068, -0.0016),
]

ARCH_A = 0.0292      # demi-largeur de l'arcade
ARCH_B = 0.0258      # profondeur de l'arcade
ARCH_Y = 0.0430
EDGE_Z = 0.0378
GUM_Z = EDGE_Z + 0.0102


def _superellipsoid(radii, exponent=3.4, rings=14, segments=20):
    phi = np.linspace(0.0, np.pi, rings)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    e = 2.0 / exponent

    sp, cp = np.sin(P), np.cos(P)
    st, ct = np.sin(T), np.cos(T)

    def p(v, k):
        return np.sign(v) * np.abs(v) ** k

    x = radii[0] * p(sp, e) * p(ct, e)
    y = radii[1] * p(sp, e) * p(st, e)
    z = radii[2] * p(cp, e)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return verts, grid_faces(rings, segments, close_cols=True)


def _tooth(index, side, name):
    half, height, recess = UPPER_TEETH[index]

    # position angulaire cumulee le long de l'arcade
    span = sum(2.0 * t[0] for t in UPPER_TEETH[:index]) + half
    t = span / ARCH_A * 0.92
    cx = side * ARCH_A * np.sin(t)
    cy = ARCH_Y + (ARCH_B + recess) * np.cos(t)
    cz = EDGE_Z + height * 0.5

    verts, faces = _superellipsoid((half, 0.0052, height * 0.5), exponent=3.2)
    # la couronne s'affine vers le bord libre et se creuse a l'arriere
    u = (verts[:, 2] + height * 0.5) / max(height, 1e-6)
    verts[:, 0] *= 0.80 + 0.20 * u
    verts[:, 1] *= 0.72 + 0.28 * u
    verts[:, 1] -= 0.0016 * (1.0 - u)

    # rotation tangente a l'arcade
    ang = -side * t
    c, s = np.cos(ang), np.sin(ang)
    rot = np.stack([verts[:, 0] * c - verts[:, 1] * s,
                    verts[:, 0] * s + verts[:, 1] * c,
                    verts[:, 2]], axis=1)
    rot += np.array([cx, cy, cz])
    return object_from_arrays(name, rot, faces)


def build_teeth():
    from .materials import teeth_material, mouth_material

    parts = []
    for side in (1.0, -1.0):
        for i in range(len(UPPER_TEETH)):
            parts.append(_tooth(i, side, f"Dent{int(side)}{i}"))
    teeth = join_objects(parts, "Dents")
    teeth.data.materials.append(teeth_material())

    # gencive : un bourrelet qui suit l'arcade, juste au-dessus des couronnes
    t = np.linspace(-1.15, 1.15, 40)
    ring = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)
    T, R = np.meshgrid(t, ring, indexing="ij")
    cx = ARCH_A * np.sin(T)
    cy = ARCH_Y + ARCH_B * np.cos(T)
    tang = np.stack([np.cos(T), -np.sin(T) * ARCH_B / ARCH_A], axis=-1)
    nx, ny = tang[..., 1], -tang[..., 0]
    norm = np.sqrt(nx ** 2 + ny ** 2)
    nx, ny = nx / norm, ny / norm
    rr = 0.0050
    verts = np.stack([cx + nx * rr * np.cos(R),
                      cy + ny * rr * np.cos(R),
                      GUM_Z + 0.0022 + rr * 0.9 * np.sin(R)], axis=-1).reshape(-1, 3)
    gum = object_from_arrays("Gencive", verts, grid_faces(len(t), len(ring)))
    gum.data.materials.append(mouth_material())

    # fond de bouche : une poche sombre qui bouche toute fuite de lumiere
    from .meshtools import uv_sphere_arrays
    v, f = uv_sphere_arrays(radius=1.0, rings=20, segments=28)
    v *= np.array([0.0330, 0.0250, 0.0175])
    v += np.array([0.0, 0.0320, 0.0355])
    bag = object_from_arrays("FondBouche", v, f)
    bag.data.materials.append(mouth_material())

    # langue, a peine visible mais elle evite le trou noir
    v, f = uv_sphere_arrays(radius=1.0, rings=18, segments=24)
    v *= np.array([0.0210, 0.0260, 0.0072])
    v += np.array([0.0, 0.0385, 0.0288])
    tongue = object_from_arrays("Langue", v, f)
    tongue.data.materials.append(mouth_material())

    return [teeth, gum, bag, tongue]
