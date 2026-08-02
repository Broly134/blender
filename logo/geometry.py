"""Contours 2D : mise a l'echelle, et conversion des traits en contours fermes.

Un `stroke` SVG n'est pas une surface : c'est une ligne avec une epaisseur. Pour
l'extruder en 3D il faut d'abord en fabriquer le contour ferme, avec des
raccords en onglet aux sommets — sinon les angles du cadre partent en biseaux
sales ou en trous.
"""

import numpy as np


def _clean(pts, eps=1e-9):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > eps:
            keep.append(i)
    return pts[keep]


def stroke_outline(pts, width, closed=False, miter_limit=6.0):
    """Contour ferme d'une polyligne epaisse, avec raccords en onglet."""
    pts = _clean(np.asarray(pts, dtype=np.float64))
    if len(pts) < 2:
        return []
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])

    h = width * 0.5
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(seg, 1e-12)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)      # normale a gauche

    offsets = np.empty_like(pts)
    offsets[0] = n[0] * h
    offsets[-1] = n[-1] * h
    for j in range(1, len(pts) - 1):
        m = n[j - 1] + n[j]
        norm = np.linalg.norm(m)
        if norm < 1e-9:                            # demi-tour
            offsets[j] = n[j] * h
            continue
        m /= norm
        cos_half = float(np.dot(m, n[j]))
        scale = h / max(cos_half, 1.0 / miter_limit)
        offsets[j] = m * scale

    left = pts + offsets
    right = pts - offsets
    if closed:
        return [(left[:-1], True), (right[:-1][::-1], True)]
    return [(np.vstack([left, right[::-1]]), True)]


def contours_of(shape):
    """Contours fermes d'un element, qu'il soit rempli ou trace."""
    out = []
    if shape.fill is not None:
        for pts, _ in shape.subpaths:
            if len(pts) >= 3:
                out.append(pts)
    if shape.stroke is not None and shape.width > 0.0:
        for pts, closed in shape.subpaths:
            for c, _ in stroke_outline(pts, shape.width, closed=closed):
                out.append(c)
    return out


def normalize(all_contours, size=1.0):
    """Centre, met a l'echelle et retourne l'axe Y (le SVG a Y vers le bas).

    Renvoie la fonction de transformation, pour l'appliquer identiquement a
    tous les elements.
    """
    pts = np.concatenate(all_contours)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) * 0.5
    span = float(max(hi - lo))
    scale = size / span if span > 0 else 1.0

    def transform(p):
        q = (np.asarray(p, dtype=np.float64) - center) * scale
        q[:, 1] *= -1.0                # Y vers le haut
        return q

    extent = (hi - lo) * scale
    return transform, extent
