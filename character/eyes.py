"""Globes oculaires, decoupe de la fente palpebrale et bord libre."""

import numpy as np

from . import anatomy as A
from .mathutil import smoothstep
from .meshtools import (object_from_arrays, uv_sphere_arrays, delete_faces_by_mask,
                        snap_boundary, extrude_boundary_inward, relax_region)


FISSURE_HALF_W = 0.01640     # de la caroncule au canthus externe
FISSURE_UP = 0.00600         # hauteur de la paupiere superieure
FISSURE_DOWN = 0.00520
FISSURE_TILT = np.radians(6.5)   # le canthus externe est plus haut


def _side_of(p):
    return 1.0 if p[0] >= 0.0 else -1.0


def _local(p, sx):
    """Coordonnees (u, w) dans le repere incline de la fente."""
    dx = (p[0] - sx * A.EYE_X) * sx
    dz = p[2] - A.EYE_LINE
    c, s = np.cos(FISSURE_TILT), np.sin(FISSURE_TILT)
    return dx * c + dz * s, -dx * s + dz * c


def _world(u, w, y, sx):
    c, s = np.cos(FISSURE_TILT), np.sin(FISSURE_TILT)
    dx = u * c - w * s
    dz = u * s + w * c
    return np.array([sx * A.EYE_X + sx * dx, y, A.EYE_LINE + dz])


def _energy(p):
    """Forme quadratique valant 1 sur le contour de la fente.

    Deux demi-ellipses raccordees : le raccord a w = 0 produit exactement les
    deux coins pointus de l'oeil.
    """
    sx = _side_of(p)
    u, w = _local(p, sx)
    h = FISSURE_UP if w >= 0.0 else FISSURE_DOWN
    return (u / FISSURE_HALF_W) ** 2 + (w / h) ** 2


def _in_eye_region(p):
    return p[1] > 0.045 and abs(p[2] - A.EYE_LINE) < 0.028


def cut_eye_openings(head):
    delete_faces_by_mask(
        head, lambda p: not (_in_eye_region(p) and _energy(p) < 1.0))

    def project(p):
        if not _in_eye_region(p):
            return None
        e = _energy(p)
        if not (0.05 < e < 4.0):
            return None
        sx = _side_of(p)
        u, w = _local(p, sx)
        s = 1.0 / max(np.sqrt(e), 1e-9)
        return _world(u * s, w * s, p[1], sx)

    snap_boundary(head, project)

    def band(p):
        if not _in_eye_region(p):
            return 0.0
        e = _energy(p)
        return float(1.0 - smoothstep(1.6, 6.0, e))

    relax_region(head, band, iterations=4, factor=0.60)

    centers = {1.0: np.array([A.EYE_X, A.EYE_Y, A.EYE_LINE]),
               -1.0: np.array([-A.EYE_X, A.EYE_Y, A.EYE_LINE])}

    def displace(p):
        sx = _side_of(p)
        c = centers[sx]
        radial = p - c
        radial = radial / max(np.linalg.norm(radial), 1e-9)
        target = c + radial * (A.EYE_R + 0.0003)
        _, w = _local(p, sx)
        out = p + (target - p) * 0.80
        out[2] -= (0.0007 if w > 0 else -0.0007)
        return out

    extrude_boundary_inward(head, _in_eye_region, displace)
    return head


def _eyeball(sx, gaze_target):
    """Globe en repere canonique : la pupille regarde +Y en local.

    On ne cuit surtout pas la rotation dans le maillage : le shader lit la
    coordonnee Object pour placer iris et pupille, il faut donc que le repere
    local reste solidaire du regard. C'est l'objet qu'on oriente.
    """
    import mathutils

    verts, faces = uv_sphere_arrays(radius=A.EYE_R, rings=52, segments=72)
    verts = verts[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])   # poles sur Y

    # bombement corneen : la cornee depasse la sphere d'environ 1,6 mm
    d = verts / A.EYE_R
    cap = smoothstep(np.cos(np.radians(35.0)), 1.0, d[:, 1])
    verts += d * (cap ** 1.5 * 0.00155)[:, None]

    obj = object_from_arrays(f"Oeil{'G' if sx > 0 else 'D'}", verts, faces)
    center = mathutils.Vector((sx * A.EYE_X, A.EYE_Y, A.EYE_LINE))
    obj.location = center
    direction = (mathutils.Vector(gaze_target) - center).normalized()
    obj.rotation_euler = direction.to_track_quat("Y", "Z").to_euler()
    return obj


def build_eyes(gaze_target=(0.012, 0.930, 0.120)):
    from .materials import eye_material
    mat = eye_material()
    objs = []
    for sx in (1.0, -1.0):
        o = _eyeball(sx, gaze_target)
        o.data.materials.append(mat)
        objs.append(o)
    return objs
