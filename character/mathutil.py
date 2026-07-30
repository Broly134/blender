"""Petits outils mathematiques partages par tous les modules de construction.

Tout est vectorise avec numpy : les maillages sont generes comme des tableaux
(N, 3) puis pousses dans Blender en une seule fois.
"""

import numpy as np


# --------------------------------------------------------------------------
# noyaux de lissage
# --------------------------------------------------------------------------

def bump(t):
    """Noyau C2 a support compact : vaut 1 en 0, 0 des que |t| >= 1."""
    t = np.clip(np.abs(np.asarray(t, dtype=np.float64)), 0.0, 1.0)
    return (1.0 - t * t) ** 3


def smoothstep(edge0, edge1, x):
    """Interpolation d'Hermite classique, saturee en dehors de [edge0, edge1]."""
    t = np.clip((np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pchip_slopes(x, y):
    """Pentes de Fritsch-Carlson : interpolation cubique sans overshoot."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


def _gaussian_smooth(values, sigma_samples):
    """Convolution gaussienne avec prolongement par les bords."""
    if sigma_samples <= 0.0:
        return values
    radius = int(max(1, round(3.0 * sigma_samples)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    k /= k.sum()
    padded = np.concatenate([
        np.full(radius, values[0]),
        values,
        np.full(radius, values[-1]),
    ])
    return np.convolve(padded, k, mode="valid")


class Profile:
    """Courbe 1D definie par des points de controle (t, valeur).

    On interpole en cubique monotone (passe exactement par les points, pas
    d'oscillation), puis on applique un tres leger flou gaussien : ca supprime
    les discontinuites de courbure qui se verraient comme des bandes sur une
    peau brillante.
    """

    def __init__(self, knots, samples=2048, smooth=0.004):
        knots = sorted(knots, key=lambda kv: kv[0])
        xs = np.array([k[0] for k in knots], dtype=np.float64)
        ys = np.array([k[1] for k in knots], dtype=np.float64)
        self.x0, self.x1 = float(xs[0]), float(xs[-1])

        grid = np.linspace(self.x0, self.x1, samples)
        m = _pchip_slopes(xs, ys)

        idx = np.clip(np.searchsorted(xs, grid) - 1, 0, len(xs) - 2)
        h = xs[idx + 1] - xs[idx]
        s = (grid - xs[idx]) / h
        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        vals = (h00 * ys[idx] + h10 * h * m[idx]
                + h01 * ys[idx + 1] + h11 * h * m[idx + 1])

        span = self.x1 - self.x0
        sigma_samples = smooth * span / (span / (samples - 1))
        self.grid = grid
        self.vals = _gaussian_smooth(vals, sigma_samples)

    def __call__(self, x):
        return np.interp(np.asarray(x, dtype=np.float64), self.grid, self.vals)


def biased_samples(count, weight_fn, lo, hi, periodic=False):
    """Repartit `count` echantillons sur [lo, hi] selon une densite donnee.

    Sert a concentrer les anneaux du maillage la ou il faut du detail
    (le visage) plutot que sur l'arriere du crane.
    """
    fine = np.linspace(lo, hi, 4096)
    w = np.asarray(weight_fn(fine), dtype=np.float64)
    w = np.maximum(w, 1e-6)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(fine))])
    cdf /= cdf[-1]
    if periodic:
        targets = np.linspace(0.0, 1.0, count, endpoint=False)
    else:
        targets = np.linspace(0.0, 1.0, count)
    return np.interp(targets, cdf, fine)


