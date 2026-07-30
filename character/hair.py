"""Pilosite : cheveux ras et crepus, barbe fournie, moustache, sourcils, cils.

Tout passe par des systemes de particules "hair" pilotes par des groupes de
sommets calcules analytiquement (implantation, densite, longueur). Le frisottis
serre vient d'un kink de type CURL a haute frequence : c'est ce qui donne la
texture d'une barbe et de cheveux afro courts.
"""

import numpy as np

from . import anatomy as A
from .mathutil import smoothstep, bump
from .meshtools import add_vertex_group, vertex_coords


def _noise(v, scale, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros(len(v))
    for _ in range(3):
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        out += np.sin(scale * (v @ k) + rng.uniform(0, 6.283))
    return out / 3.0


# ---------------------------------------------------------------------------
# implantations
# ---------------------------------------------------------------------------

def scalp_weights(v):
    """Cuir chevelu : dense partout sauf sur le visage, avec des golfes frontaux."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    edge = _noise(v, 120.0, 21) * 0.010

    # ligne frontale : golfes temporaux marques, puis favoris devant l'oreille
    ax = np.abs(x)
    hairline = np.interp(ax, [0.000, 0.030, 0.048, 0.060, 0.072, 0.090],
                         [0.178, 0.180, 0.192, 0.160, 0.104, 0.098])
    hairline = np.where(y > 0.02, hairline, 0.055)
    w = smoothstep(-0.012, 0.014, z - hairline + edge)

    # pas de cheveux sur les oreilles ni juste devant
    w *= smoothstep(0.062, 0.074, np.abs(x)) * 0.0 + 1.0
    w *= 1.0 - 0.95 * np.exp(-0.5 * (((np.abs(x) - A.EAR_X) / 0.017) ** 2
                                     + ((y - A.EAR_Y) / 0.024) ** 2
                                     + ((z - A.EAR_Z) / 0.036) ** 2))
    # descend sur la nuque
    nape = smoothstep(-0.055, -0.015, z) * smoothstep(0.0, -0.03, y)
    w = np.maximum(w * smoothstep(-0.02, 0.02, z - 0.03), nape * 0.85)
    return np.clip(w, 0.0, 1.0)


def beard_weights(v):
    """Barbe pleine : machoire, menton, joues jusqu'a mi-hauteur, haut du cou."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    edge = _noise(v, 95.0, 5) * 0.009

    # ligne haute de la barbe : du favori vers la commissure
    line = 0.021 + 0.66 * np.abs(x)
    line = np.minimum(line, 0.072)
    w = smoothstep(0.010, -0.012, z - line + edge)

    # devant seulement, et pas trop bas dans le cou
    w *= smoothstep(-0.010, 0.030, y)
    w *= smoothstep(-0.078, -0.042, z + edge * 1.6)

    # ni sur les levres ni sur le bord des narines
    lips = np.exp(-0.5 * (((x) / 0.030) ** 2
                          + ((z - A.STOMION) / 0.0105) ** 2
                          + ((y - 0.086) / 0.030) ** 2))
    w *= 1.0 - np.clip(lips * 1.9, 0.0, 1.0)
    nose = np.exp(-0.5 * (((x) / 0.030) ** 2 + ((z - A.SUBNASALE) / 0.014) ** 2))
    w *= 1.0 - np.clip(nose * 1.6, 0.0, 1.0)

    # favoris qui rejoignent les cheveux devant l'oreille
    side = np.exp(-0.5 * (((np.abs(x) - 0.070) / 0.009) ** 2
                          + ((z - 0.080) / 0.016) ** 2
                          + ((y - 0.000) / 0.024) ** 2))
    w = np.maximum(w, side * 0.80)
    return np.clip(w, 0.0, 1.0)


def moustache_weights(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    edge = _noise(v, 160.0, 12) * 0.0015
    w = np.exp(-0.5 * (((x) / 0.0225) ** 2
                       + ((z - (A.STOMION + 0.0125)) / 0.0062) ** 2))
    # les pointes rejoignent la barbe aux commissures
    w = np.maximum(w, np.exp(-0.5 * (((np.abs(x) - 0.0250) / 0.0075) ** 2
                                     + ((z - (A.STOMION + 0.0040)) / 0.0075) ** 2)))
    w *= smoothstep(0.030, 0.055, y)
    w *= smoothstep(A.STOMION + 0.0035, A.STOMION + 0.0070, z + edge)
    return np.clip(w * 1.4, 0.0, 1.0)


def brow_weights(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax = np.abs(x)
    # l'arc du sourcil monte legerement vers l'exterieur puis retombe
    arc = A.BROW - 0.0015 + 0.055 * ax - 1.35 * np.clip(ax - 0.028, 0.0, 1.0) ** 1.4
    w = np.exp(-0.5 * ((z - arc) / 0.0040) ** 2)
    w *= smoothstep(0.005, 0.012, ax) * smoothstep(0.047, 0.037, ax)
    w *= smoothstep(0.030, 0.055, y)
    return np.clip(w * 1.5, 0.0, 1.0)


def lash_weights(v, upper=True):
    from .eyes import FISSURE_HALF_W, FISSURE_UP, FISSURE_DOWN, _local, _side_of
    w = np.zeros(len(v))
    for i, p in enumerate(v):
        if p[1] < 0.050 or abs(p[2] - A.EYE_LINE) > 0.020:
            continue
        sx = _side_of(p)
        u, ww = _local(p, sx)
        if abs(u) > FISSURE_HALF_W * 1.02:
            continue
        h = FISSURE_UP if ww >= 0 else FISSURE_DOWN
        e = (u / FISSURE_HALF_W) ** 2 + (ww / h) ** 2
        if 0.55 < e < 1.35 and ((ww > 0) == upper):
            w[i] = 1.0
    return w


# ---------------------------------------------------------------------------
# systemes de particules
# ---------------------------------------------------------------------------

def add_hair(obj, name, group, material_index, *, count, length,
             children=14, kink_amp=0.0012, kink_freq=9.0, roughness=0.0016,
             clump=-0.35, length_random=0.45, quality=1.0, tip=0.00006,
             root=0.00016, brownian=0.0):
    mod = obj.modifiers.new(name, "PARTICLE_SYSTEM")
    psys = obj.particle_systems[-1]
    psys.name = name
    ps = psys.settings
    ps.name = f"Reglages{name}"
    ps.type = "HAIR"
    ps.count = max(16, int(count * quality))
    ps.hair_length = length
    ps.hair_step = 6
    ps.use_advanced_hair = True
    ps.emit_from = "FACE"
    ps.distribution = "RAND"
    ps.use_even_distribution = True
    ps.material = material_index + 1

    ps.child_type = "INTERPOLATED"
    ps.child_percent = max(2, int(children * 0.35))
    ps.rendered_child_count = max(2, int(children * quality))
    ps.child_radius = 0.0013
    ps.child_roundness = 1.0
    ps.clump_factor = clump
    ps.roughness_1 = roughness
    ps.roughness_1_size = 0.006
    ps.roughness_2 = roughness * 0.7
    ps.roughness_2_size = 0.02
    ps.roughness_endpoint = roughness * 0.5

    ps.kink = "CURL"
    ps.kink_amplitude = kink_amp
    ps.kink_frequency = kink_freq
    ps.kink_shape = 0.25
    ps.kink_flat = 0.35

    ps.brownian_factor = brownian
    ps.factor_random = length_random * length * 0.6

    for attr, value in (("root_radius", root), ("tip_radius", tip),
                        ("radius_scale", 1.0), ("shape", 0.15)):
        if hasattr(ps, attr):
            setattr(ps, attr, value)

    psys.vertex_group_density = group
    psys.vertex_group_length = group
    return psys


def groom(head, ears, quality=1.0):
    """Pose tous les systemes de poils sur la tete (et les oreilles pour le duvet)."""
    from .materials import hair_material

    v = vertex_coords(head)
    groups = {
        "scalp": scalp_weights(v),
        "beard": beard_weights(v),
        "moustache": moustache_weights(v),
        "brows": brow_weights(v),
        "lash_up": lash_weights(v, upper=True),
        "lash_dn": lash_weights(v, upper=False),
    }
    for name, w in groups.items():
        add_vertex_group(head, name, w)

    # trois nuances de noir : cheveux, barbe (plus chaude), sourcils
    mats = [
        hair_material("PoilCheveux", color=(0.0052, 0.0034, 0.0024),
                      roughness=0.32, radial=0.48),
        hair_material("PoilBarbe", color=(0.0068, 0.0042, 0.0029),
                      roughness=0.38, radial=0.55),
        hair_material("PoilSourcil", color=(0.0044, 0.0028, 0.0020),
                      roughness=0.30, radial=0.42),
    ]
    for m in mats:
        head.data.materials.append(m)
    base = len(head.data.materials) - len(mats)

    add_hair(head, "Cheveux", "scalp", base + 0,
             count=int(11000), length=0.0088, children=30,
             kink_amp=0.0012, kink_freq=11.0, roughness=0.0014,
             clump=-0.40, length_random=0.30, quality=quality,
             root=0.00012, tip=0.00007)

    add_hair(head, "Barbe", "beard", base + 1,
             count=int(12000), length=0.0094, children=38,
             kink_amp=0.0012, kink_freq=8.0, roughness=0.0017,
             clump=-0.68, length_random=0.50, quality=quality,
             root=0.00014, tip=0.00008)

    add_hair(head, "Moustache", "moustache", base + 1,
             count=int(2600), length=0.0082, children=24,
             kink_amp=0.0009, kink_freq=7.0, roughness=0.0010,
             clump=-0.60, length_random=0.30, quality=quality,
             root=0.00013, tip=0.00007)

    add_hair(head, "Sourcils", "brows", base + 2,
             count=int(1700), length=0.0052, children=16,
             kink_amp=0.0005, kink_freq=5.0, roughness=0.0007,
             clump=-0.62, length_random=0.30, quality=quality,
             root=0.00012, tip=0.00005)

    add_hair(head, "CilsHaut", "lash_up", base + 2,
             count=int(320), length=0.0072, children=6,
             kink_amp=0.0006, kink_freq=2.0, roughness=0.0006,
             clump=-0.5, quality=quality, root=0.00011, tip=0.00004)

    add_hair(head, "CilsBas", "lash_dn", base + 2,
             count=int(240), length=0.0038, children=5,
             kink_amp=0.0004, kink_freq=2.0, roughness=0.0005,
             clump=-0.5, quality=quality, root=0.00009, tip=0.00004)

    return head
