"""Variations regionales de la peau, ecrites dans des attributs de sommets.

Plutot que de peindre des textures a la main, on calcule analytiquement les
zones que tout portrait realiste demande : levres plus violacees, cerne
periorbitaire, ailes du nez et oreilles plus rouges (la ou la peau est fine et
tres vascularisee), zones grasses du front et du nez, assombrissement de la
zone de barbe. Le shader n'a plus qu'a lire ces attributs.
"""

import numpy as np

from . import anatomy as A
from .mathutil import bump, smoothstep


def _blob(v, center, radii, power=1.0, mirror=False):
    total = np.zeros(len(v))
    centers = [center]
    if mirror:
        centers.append((-center[0], center[1], center[2]))
    for c in centers:
        d = (v - np.asarray(c)) / np.asarray(radii)
        total = np.maximum(total, bump(np.sqrt(np.sum(d * d, axis=1))) ** power)
    return total


def _noise(v, scale, seed):
    """Bruit doux et deterministe (somme de sinusoides 3D)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(v))
    for _ in range(4):
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        phase = rng.uniform(0, 6.283)
        out += np.sin(scale * (v @ k) + phase)
    return out / 4.0


def compute(v):
    """Renvoie (tint RGB, oil, sss) pour un tableau de sommets (N, 3)."""
    n = len(v)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    front = smoothstep(-0.02, 0.04, y)

    tint = np.ones((n, 3))
    oil = np.full(n, 0.12)
    sss = np.full(n, 0.12)

    # --- levres ------------------------------------------------------------
    lips = _blob(v, (0.0, 0.083, A.STOMION + 0.0005), (0.0330, 0.0180, 0.0125), 0.85)
    lips *= front
    tint[:, 0] *= 1.0 + 0.22 * lips
    tint[:, 1] *= 1.0 - 0.34 * lips
    tint[:, 2] *= 1.0 - 0.36 * lips
    sss += 0.30 * lips
    oil += 0.42 * lips

    # bord interne des levres, franchement rouge
    rim = _blob(v, (0.0, 0.072, A.STOMION), (0.0210, 0.0120, 0.0058), 1.6)
    tint[:, 0] *= 1.0 + 0.45 * rim
    tint[:, 1] *= 1.0 - 0.55 * rim
    tint[:, 2] *= 1.0 - 0.55 * rim

    # --- cerne periorbitaire ----------------------------------------------
    orbit = _blob(v, (A.EYE_X, 0.070, A.EYE_LINE - 0.0045),
                  (0.0290, 0.0260, 0.0175), 0.75, mirror=True) * front
    tint[:, 0] *= 1.0 - 0.24 * orbit
    tint[:, 1] *= 1.0 - 0.26 * orbit
    tint[:, 2] *= 1.0 - 0.24 * orbit
    sss += 0.14 * orbit

    # --- nez : ailes et pointe, peau fine et rouge -------------------------
    nose = _blob(v, (0.0, 0.098, A.NOSE_TIP - 0.0025), (0.0290, 0.0200, 0.0150), 0.9) * front
    tint[:, 0] *= 1.0 + 0.06 * nose
    tint[:, 1] *= 1.0 - 0.02 * nose
    tint[:, 2] *= 1.0 - 0.03 * nose
    sss += 0.26 * nose

    # --- oreilles ----------------------------------------------------------
    ear = _blob(v, (A.EAR_X + 0.008, A.EAR_Y, A.EAR_Z), (0.0300, 0.0330, 0.0400),
                0.8, mirror=True)
    tint[:, 0] *= 1.0 + 0.16 * ear
    tint[:, 1] *= 1.0 - 0.06 * ear
    tint[:, 2] *= 1.0 - 0.09 * ear
    sss += 0.38 * ear

    # --- zone de barbe : la peau y est visuellement plus sombre ------------
    beard_line = 0.038 + 0.95 * np.abs(x)
    beard = smoothstep(0.012, -0.010, z - beard_line) * front
    beard *= smoothstep(-0.075, -0.035, z)
    beard = np.maximum(beard - lips * 0.9, 0.0)
    tint[:, 0] *= 1.0 - 0.15 * beard
    tint[:, 1] *= 1.0 - 0.13 * beard
    tint[:, 2] *= 1.0 - 0.10 * beard

    # --- reliefs gras : front, dos du nez, pommettes, menton ---------------
    oil += 0.85 * _blob(v, (0.0, 0.084, 0.1560), (0.0520, 0.0280, 0.0290), 0.8) * front
    oil += 0.90 * _blob(v, (0.0, 0.092, 0.0980), (0.0170, 0.0250, 0.0330), 0.8) * front
    oil += 0.55 * _blob(v, (0.0540, 0.066, 0.1000), (0.0270, 0.0300, 0.0230), 0.9,
                        mirror=True) * front
    oil += 0.45 * _blob(v, (0.0, 0.078, 0.0120), (0.0260, 0.0220, 0.0160), 0.9) * front
    oil += 0.60 * _blob(v, (A.EYE_X, 0.074, A.EYE_LINE + 0.0080),
                        (0.0230, 0.0200, 0.0080), 0.9, mirror=True) * front

    # --- irregularites : personne n'a un teint uniforme --------------------
    blotch = _noise(v, 55.0, 3)
    tint *= (1.0 + 0.030 * blotch)[:, None]
    oil *= 1.0 + 0.25 * _noise(v, 90.0, 8)

    # le cou et la nuque sont un peu plus sombres et moins gras
    neck = smoothstep(0.010, -0.045, z)
    tint *= (1.0 - 0.10 * neck)[:, None]
    oil *= 1.0 - 0.55 * neck

    return np.clip(tint, 0.0, 2.0), np.clip(oil, 0.0, 1.0), np.clip(sss, 0.0, 1.0)


def paint(obj):
    """Ecrit les trois attributs sur le maillage."""
    from .meshtools import vertex_coords
    mesh = obj.data
    v = vertex_coords(obj)
    tint, oil, sss = compute(v)

    col = mesh.color_attributes.new(name="skin_tint", type="FLOAT_COLOR",
                                    domain="POINT")
    rgba = np.concatenate([tint, np.ones((len(v), 1))], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel())

    for name, values in (("skin_oil", oil), ("skin_sss", sss)):
        attr = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
        attr.data.foreach_set("value", values.astype(np.float32))

    mesh.update()
    return obj
