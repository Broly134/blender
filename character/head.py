"""Construction de la tete : loft anatomique puis sculpture des traits.

Le maillage de base est un loft d'anneaux super-elliptiques (un quadrant
avant + un quadrant arriere avec des exposants differents) ; ensuite on
applique une pile de deformations locales, exactement comme des coups de
brosse "grab" en sculpture, mais definies analytiquement.
"""

import numpy as np

from . import anatomy as A
from .mathutil import bump, smoothstep, biased_samples, Profile
from .meshtools import object_from_arrays, grid_faces, cap_ring


N_RINGS = 190
N_SEG = 176


# ---------------------------------------------------------------------------
# maillage de base
# ---------------------------------------------------------------------------

def _ring_density(z):
    """Plus d'anneaux sur le visage (yeux -> menton) que sur le crane."""
    d = np.ones_like(z) * 0.55
    d += 1.55 * np.exp(-0.5 * ((z - 0.075) / 0.055) ** 2)   # nez / bouche
    d += 1.10 * np.exp(-0.5 * ((z - 0.115) / 0.030) ** 2)   # yeux
    d += 0.45 * np.exp(-0.5 * ((z - 0.010) / 0.030) ** 2)   # menton
    return d


def _seg_density(theta):
    """Plus de segments sur la face avant."""
    front = np.sin(theta)
    return 0.55 + 1.35 * np.clip(front, 0.0, 1.0) ** 1.4


def base_surface():
    zs = biased_samples(N_RINGS, _ring_density, A.NECK_BASE, A.VERTEX)
    th = biased_samples(N_SEG, _seg_density, 0.0, 2.0 * np.pi, periodic=True)

    Z, T = np.meshgrid(zs, th, indexing="ij")
    w = A.HALF_WIDTH(Z)
    f = A.FRONT(Z)
    b = A.BACK(Z)
    nf = A.EXP_FRONT(Z)
    nb = A.EXP_BACK(Z)

    c, s = np.cos(T), np.sin(T)
    front_side = s >= 0.0
    n = np.where(front_side, nf, nb)
    depth = np.where(front_side, f, b)

    e = 2.0 / n
    x = w * np.sign(c) * np.abs(c) ** e
    y = depth * np.sign(s) * np.abs(s) ** e

    verts = np.stack([x, y, Z], axis=-1).reshape(-1, 3)
    return verts, zs, th


# ---------------------------------------------------------------------------
# outils de sculpture
# ---------------------------------------------------------------------------

def _front_gate(v, start=-0.005, end=0.030):
    return smoothstep(start, end, v[:, 1])


def add_bump(v, center, radii, amount, direction="Y", power=1.0,
             mirror=False, gate=None):
    """Deplace les sommets d'une zone ellipsoidale, facon brosse 'grab'."""
    centers = [center]
    if mirror:
        centers.append((-center[0], center[1], center[2]))

    for ci, c in enumerate(centers):
        d = (v - np.asarray(c)) / np.asarray(radii)
        t = np.sqrt(np.sum(d * d, axis=1))
        wgt = bump(t) ** power
        if gate is not None:
            wgt = wgt * gate
        if isinstance(direction, str):
            if direction == "Y":
                dirs = np.tile(np.array([0.0, 1.0, 0.0]), (len(v), 1))
            elif direction == "Z":
                dirs = np.tile(np.array([0.0, 0.0, 1.0]), (len(v), 1))
            elif direction == "XY":       # radial, vers l'exterieur
                dirs = np.stack([v[:, 0], v[:, 1], np.zeros(len(v))], axis=1)
                dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
            elif direction == "X":
                sign = -1.0 if ci == 1 else 1.0
                dirs = np.tile(np.array([sign, 0.0, 0.0]), (len(v), 1))
            else:
                raise ValueError(direction)
        else:
            dvec = np.asarray(direction, dtype=np.float64)
            if ci == 1:
                dvec = dvec * np.array([-1.0, 1.0, 1.0])
            dirs = np.tile(dvec / np.linalg.norm(dvec), (len(v), 1))
        v += dirs * (wgt * amount)[:, None]
    return v


# ---------------------------------------------------------------------------
# traits du visage
# ---------------------------------------------------------------------------

def sculpt_nose(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    gate = _front_gate(v, 0.010, 0.045)

    ridge = A.NOSE_RIDGE(np.clip(z, 0.0555, 0.1290))
    ridge = np.where((z > 0.0545) & (z < 0.1300), ridge, 0.0)
    half = np.maximum(A.NOSE_HALF(np.clip(z, 0.0545, 0.1290)), 1e-4)

    u = np.abs(x) / (half * 1.42)
    across = np.where(u < 1.0, np.clip(1.0 - u * u, 0.0, 1.0) ** 1.25, 0.0)
    v[:, 1] += ridge * across * gate

    # ailes du nez : deux volumes charnus qui debordent lateralement
    add_bump(v, (0.0205, 0.0895, A.ALAR - 0.0020), (0.0140, 0.0190, 0.0135),
             0.0044, direction=(0.68, 0.72, -0.10), mirror=True, gate=gate)
    # sillon alaire (le pli qui detache l'aile de la joue)
    add_bump(v, (0.0282, 0.0845, A.ALAR - 0.0045), (0.0068, 0.0125, 0.0115),
             -0.0026, direction=(0.50, 0.86, 0.0), mirror=True, gate=gate)
    # pointe charnue, une seule masse arrondie
    add_bump(v, (0.0000, 0.0985, A.NOSE_TIP + 0.0008), (0.0125, 0.0180, 0.0115),
             0.0032, direction=(0, 0.99, 0.10), gate=gate)
    # sous-cloison
    add_bump(v, (0.0000, 0.0930, A.SUBNASALE - 0.0010), (0.0055, 0.0150, 0.0080),
             0.0022, direction="Y", gate=gate)
    # narines : la matiere est repoussee en arriere ET vers le haut, ce qui
    # cree le vrai surplomb sous la pointe du nez
    add_bump(v, (0.0098, 0.0900, A.SUBNASALE - 0.0048), (0.0062, 0.0125, 0.0056),
             0.0105, direction=(0.06, -0.56, 0.83), mirror=True, gate=gate)
    # rebord charnu de la narine
    add_bump(v, (0.0170, 0.0885, A.SUBNASALE - 0.0035), (0.0068, 0.0120, 0.0062),
             0.0022, direction=(0.30, 0.60, -0.74), mirror=True, gate=gate)
    # creux de la racine du nez
    add_bump(v, (0.0000, 0.0870, A.NASION + 0.0015), (0.0165, 0.0160, 0.0090),
             -0.0034, direction="Y", gate=gate)
    return v


def sculpt_eyes(v):
    """Conforme la paupiere au globe oculaire puis ajoute les plis."""
    gate = _front_gate(v, -0.010, 0.030)
    c = np.array([A.EYE_X, A.EYE_Y, A.EYE_LINE])
    lid_r = A.EYE_R + 0.0019

    for sx in (1.0, -1.0):
        cc = c * np.array([sx, 1.0, 1.0])
        d = (v - cc) / np.array([0.0300, 0.0340, 0.0235])
        t = np.sqrt(np.sum(d * d, axis=1))
        w = bump(t) ** 0.85 * gate
        rel = v - cc
        dist = np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
        # seulement devant le globe
        w = w * smoothstep(-0.006, 0.008, rel[:, 1])
        target = cc + rel / dist[:, None] * lid_r
        v += (target - v) * w[:, None]

    # arcade sourciliere marquee
    add_bump(v, (0.0300, 0.0870, A.BROW + 0.0010), (0.0350, 0.0240, 0.0135),
             0.0098, direction=(0.12, 0.98, 0.14), mirror=True, gate=gate)
    add_bump(v, (0.0000, 0.0880, A.GLABELLA + 0.0010), (0.0165, 0.0210, 0.0130),
             0.0044, direction="Y", gate=gate)
    # leger retrait juste au-dessus de l'arcade
    add_bump(v, (0.0250, 0.0850, A.BROW + 0.0195), (0.0380, 0.0230, 0.0130),
             -0.0014, direction="Y", mirror=True, gate=gate)
    # pli de la paupiere superieure
    add_bump(v, (A.EYE_X, 0.0790, A.EYE_LINE + 0.0110), (0.0250, 0.0180, 0.0042),
             -0.0022, direction="Y", mirror=True, gate=gate)
    # poche / rebord de la paupiere inferieure
    add_bump(v, (A.EYE_X + 0.0015, 0.0770, A.EYE_LINE - 0.0135), (0.0245, 0.0170, 0.0065),
             0.0026, direction="Y", mirror=True, gate=gate)
    # coin interne (caroncule) enfonce
    add_bump(v, (0.0175, 0.0745, A.EYE_LINE - 0.0018), (0.0075, 0.0140, 0.0075),
             -0.0026, direction="Y", mirror=True, gate=gate)
    return v


def sculpt_mouth(v):
    gate = _front_gate(v, 0.005, 0.035)
    st = A.STOMION

    # levre superieure : tubercule central + deux lobes (arc de Cupidon)
    add_bump(v, (0.0000, 0.0790, st + 0.0048), (0.0090, 0.0155, 0.0058),
             0.0090, direction=(0, 0.98, -0.18), gate=gate)
    add_bump(v, (0.0135, 0.0782, st + 0.0042), (0.0152, 0.0155, 0.0056),
             0.0074, direction=(0.10, 0.99, -0.10), mirror=True, gate=gate)
    # levre inferieure, plus pleine et plus avancee
    add_bump(v, (0.0000, 0.0782, st - 0.0072), (0.0190, 0.0165, 0.0076),
             0.0098, direction=(0, 0.99, 0.06), gate=gate)
    add_bump(v, (0.0132, 0.0778, st - 0.0060), (0.0168, 0.0158, 0.0068),
             0.0062, direction="Y", mirror=True, gate=gate)
    # commissures rentrantes
    add_bump(v, (A.MOUTH_HALF + 0.0012, 0.0762, st - 0.0012),
             (0.0108, 0.0135, 0.0088), -0.0046,
             direction=(0.32, 0.95, 0.0), mirror=True, gate=gate)
    # ligne de fermeture des levres
    add_bump(v, (0.0000, 0.0862, st), (0.0315, 0.0140, 0.0026),
             -0.0044, direction="Y", gate=gate)
    # sillons qui cernent le vermillon
    add_bump(v, (0.0000, 0.0800, st + 0.0108), (0.0245, 0.0150, 0.0038),
             -0.0024, direction="Y", gate=gate)
    add_bump(v, (0.0000, 0.0796, st - 0.0148), (0.0250, 0.0150, 0.0045),
             -0.0030, direction="Y", gate=gate)
    # philtrum : deux cretes et sa gouttiere
    add_bump(v, (0.0000, 0.0800, st + 0.0175), (0.0040, 0.0130, 0.0072),
             -0.0026, direction="Y", gate=gate)
    add_bump(v, (0.0063, 0.0800, st + 0.0170), (0.0042, 0.0130, 0.0078),
             0.0024, direction="Y", mirror=True, gate=gate)
    # sillon naso-genien
    add_bump(v, (0.0312, 0.0790, 0.0530), (0.0080, 0.0140, 0.0210),
             -0.0021, direction=(0.42, 0.90, 0.10), mirror=True, gate=gate)
    # sillon mento-labial + menton
    add_bump(v, (0.0000, 0.0778, A.SULCUS), (0.0275, 0.0175, 0.0072),
             -0.0042, direction="Y", gate=gate)
    add_bump(v, (0.0000, 0.0770, 0.0115), (0.0270, 0.0200, 0.0155),
             0.0058, direction=(0, 0.97, -0.20), gate=gate)
    add_bump(v, (0.0175, 0.0700, 0.0050), (0.0110, 0.0200, 0.0120),
             -0.0018, direction="Y", mirror=True, gate=gate)
    return v


def sculpt_structure(v):
    """Volumes generaux : pommettes, temples, masseter, occiput."""
    gate_front = _front_gate(v, -0.010, 0.030)

    # pommettes hautes et larges
    add_bump(v, (0.0545, 0.0650, 0.0995), (0.0290, 0.0360, 0.0225),
             0.0068, direction=(0.60, 0.79, 0.12), mirror=True)
    # creux sous la pommette
    add_bump(v, (0.0555, 0.0510, 0.0665), (0.0210, 0.0300, 0.0175),
             -0.0018, direction=(0.72, 0.69, 0.0), mirror=True)
    # masseter / machoire carree
    add_bump(v, (0.0600, 0.0200, 0.0450), (0.0250, 0.0350, 0.0250),
             0.0072, direction=(0.92, 0.38, 0.0), mirror=True)
    # bord inferieur de la mandibule, du menton au gonion
    add_bump(v, (0.0300, 0.0560, 0.0075), (0.0230, 0.0300, 0.0110),
             0.0038, direction=(0.45, 0.72, -0.53), mirror=True)
    add_bump(v, (0.0520, 0.0320, 0.0230), (0.0230, 0.0300, 0.0140),
             0.0040, direction=(0.72, 0.42, -0.55), mirror=True)
    # creux sous la mandibule, qui detache la machoire du cou
    add_bump(v, (0.0330, 0.0350, -0.0140), (0.0300, 0.0330, 0.0165),
             -0.0042, direction=(0.35, 0.55, -0.76), mirror=True)
    # temporal legerement creuse
    add_bump(v, (0.0700, 0.0430, 0.1420), (0.0250, 0.0330, 0.0270),
             -0.0030, direction="XY", mirror=True)
    # bosses frontales
    add_bump(v, (0.0245, 0.0790, 0.1560), (0.0300, 0.0260, 0.0230),
             0.0026, direction="Y", mirror=True, gate=gate_front)
    # leger aplatissement du front au-dessus des arcades
    add_bump(v, (0.0000, 0.0830, 0.1450), (0.0430, 0.0260, 0.0200),
             -0.0012, direction="Y", gate=gate_front)
    # occiput
    add_bump(v, (0.0000, -0.0980, 0.1420), (0.0620, 0.0330, 0.0430),
             0.0032, direction=(0, -1, 0))
    # nuque
    add_bump(v, (0.0000, -0.0680, 0.0180), (0.0430, 0.0250, 0.0330),
             -0.0028, direction=(0, -1, 0))
    # trapezes qui montent vers la nuque
    add_bump(v, (0.0430, -0.0500, -0.1150), (0.0450, 0.0500, 0.0520),
             0.0075, direction=(0.55, -0.60, -0.58), mirror=True)
    # pomme d'Adam
    add_bump(v, (0.0000, 0.0230, -0.0330), (0.0110, 0.0180, 0.0180),
             0.0034, direction="Y", gate=_front_gate(v, -0.02, 0.01))
    # sterno-cleido-mastoidiens
    add_bump(v, (0.0330, 0.0130, -0.0620), (0.0180, 0.0300, 0.0480),
             0.0030, direction=(0.60, 0.80, 0.0), mirror=True)
    return v


def add_asymmetry_and_noise(v, seed=7):
    """Casse la symetrie parfaite : c'est ce qui trahit le plus une tete CG."""
    rng = np.random.default_rng(seed)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    # leger decalage global du visage
    v[:, 0] += 0.0012 * smoothstep(-0.02, 0.16, z) * smoothstep(0.0, 0.05, y)
    # une narine un peu plus haute, une commissure plus basse
    add_bump(v, (0.0130, 0.0900, A.SUBNASALE), (0.0130, 0.0160, 0.0110),
             0.0011, direction="Z")
    add_bump(v, (-A.MOUTH_HALF, 0.0790, A.STOMION), (0.0130, 0.0140, 0.0110),
             -0.0016, direction="Z")
    add_bump(v, (0.0300, 0.0850, A.BROW), (0.0260, 0.0200, 0.0110),
             0.0013, direction="Z")

    # bruit basse frequence, dans le repere de la tete
    def wobble(freq, amp, phase):
        return amp * (np.sin(freq * (x * 1.7 + phase[0]))
                      * np.sin(freq * (y * 1.3 + phase[1]))
                      * np.sin(freq * (z + phase[2])))

    disp = np.zeros(len(v))
    for freq, amp in ((37.0, 0.00085), (73.0, 0.00040), (131.0, 0.00016)):
        disp += wobble(freq, amp, rng.uniform(0, 6.28, 3))

    nrm = np.stack([x, y * 0.85, (z - 0.10) * 0.55], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    v += nrm * disp[:, None]
    return v


# ---------------------------------------------------------------------------
# assemblage
# ---------------------------------------------------------------------------

def build_head(name="Tete"):
    verts, zs, th = base_surface()

    sculpt_structure(verts)
    sculpt_nose(verts)
    sculpt_eyes(verts)
    sculpt_mouth(verts)
    add_asymmetry_and_noise(verts)

    faces = grid_faces(N_RINGS, N_SEG, close_cols=True)
    n = len(verts)
    faces.append(cap_ring(n - N_SEG, N_SEG))             # sommet du crane
    faces.append(cap_ring(0, N_SEG, flip=True))          # base du cou

    obj = object_from_arrays(name, verts, faces)
    return obj


MOUTH_OPEN_HALF_W = 0.01620
MOUTH_OPEN_UP = 0.00330
MOUTH_OPEN_DOWN = 0.00420
MOUTH_OPEN_Z = A.STOMION - 0.0004


def _mouth_energy(p):
    u = p[0] / MOUTH_OPEN_HALF_W
    w = p[2] - MOUTH_OPEN_Z
    h = MOUTH_OPEN_UP if w >= 0.0 else MOUTH_OPEN_DOWN
    return u * u + (w / h) ** 2


def _in_mouth_region(p):
    return p[1] > 0.055 and abs(p[2] - MOUTH_OPEN_Z) < 0.014


def open_mouth(obj):
    """Entrouvre la bouche et cree le bord interne des levres."""
    from .meshtools import (delete_faces_by_mask, snap_boundary,
                            extrude_boundary_inward, relax_region)

    delete_faces_by_mask(
        obj, lambda p: not (_in_mouth_region(p) and _mouth_energy(p) < 1.0))

    def project(p):
        if not _in_mouth_region(p):
            return None
        e = _mouth_energy(p)
        if not (0.05 < e < 4.5):
            return None
        s = 1.0 / max(np.sqrt(e), 1e-9)
        w = (p[2] - MOUTH_OPEN_Z) * s
        return np.array([p[0] * s, p[1], MOUTH_OPEN_Z + w])

    snap_boundary(obj, project)

    def band(p):
        if not _in_mouth_region(p):
            return 0.0
        return float(1.0 - smoothstep(1.6, 6.5, _mouth_energy(p)))

    relax_region(obj, band, iterations=4, factor=0.60)

    def displace(p):
        up = 1.0 if p[2] > MOUTH_OPEN_Z else -1.0
        out = p.copy()
        out[1] -= 0.0092
        out[2] += up * 0.0022
        out[0] *= 0.88
        return out

    extrude_boundary_inward(obj, _in_mouth_region, displace)
    return obj
