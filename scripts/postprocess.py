"""Retouche photographique appliquee au rendu.

Le compositeur de Blender 5 exige un contexte GPU : en rendu headless il rend
une image vide. On refait donc les quelques defauts d'objectif qui manquent a
une image de synthese, directement sur les pixels : voile lumineux, aberration
chromatique, vignetage et grain.

    bpyenv/bin/python scripts/postprocess.py renders/portrait.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    import bpy
    img = bpy.data.images.load(os.path.abspath(path))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[::-1], w, h          # ligne 0 = haut de l'image


def _save(px, path, w, h):
    import bpy
    out = bpy.data.images.new("post", width=w, height=h, alpha=True)
    out.pixels = px[::-1].ravel().tolist()
    out.file_format = "PNG"
    out.filepath_raw = os.path.abspath(path)
    out.save()
    bpy.data.images.remove(out)


def _blur(a, sigma):
    radius = int(max(1, round(3.0 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    pad = ((radius, radius), (0, 0)) if a.ndim == 2 else ((radius, radius), (0, 0), (0, 0))
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0,
                              np.pad(a, pad, mode="edge"))[radius:-radius]
    pad = ((0, 0), (radius, radius)) if a.ndim == 2 else ((0, 0), (radius, radius), (0, 0))
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1,
                              np.pad(out, pad, mode="edge"))[:, radius:-radius]
    return out


def _radial_shift(channel, scale):
    """Rechantillonne un canal avec un leger grossissement autour du centre."""
    h, w = channel.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    sy = np.clip(cy + (yy - cy) / scale, 0, h - 1)
    sx = np.clip(cx + (xx - cx) / scale, 0, w - 1)
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (sy - y0)[..., None][..., 0], (sx - x0)[..., None][..., 0]
    top = channel[y0, x0] * (1 - fx) + channel[y0, x1] * fx
    bot = channel[y1, x0] * (1 - fx) + channel[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def process(px, bloom=0.085, bloom_sigma=0.018, chroma=0.0016,
            vignette=0.16, grain=0.0055, seed=5):
    h, w = px.shape[:2]
    rgb = px[..., :3].astype(np.float32)

    # --- voile lumineux : les hautes lumieres debordent -------------------
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    highs = np.clip(luma - 0.72, 0.0, None)[..., None] * rgb
    rgb = rgb + bloom * _blur(highs, bloom_sigma * h)

    # --- aberration chromatique laterale ----------------------------------
    if chroma:
        rgb = np.stack([_radial_shift(rgb[..., 0], 1.0 + chroma),
                        rgb[..., 1],
                        _radial_shift(rgb[..., 2], 1.0 - chroma)], axis=-1)

    # --- vignetage ---------------------------------------------------------
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - (w - 1) / 2) / (w / 2)) ** 2
                + ((yy - (h - 1) / 2) / (h / 2)) ** 2) / np.sqrt(2.0)
    rgb *= (1.0 - vignette * r ** 2.2)[..., None]

    # --- grain : plus visible dans les demi-tons que dans les noirs --------
    rng = np.random.default_rng(seed)
    l2 = np.clip(rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0, 1)
    amp = grain * (0.35 + 0.65 * np.sqrt(l2))
    rgb += rng.normal(0.0, 1.0, rgb.shape).astype(np.float32) * amp[..., None]

    out = px.copy()
    out[..., :3] = np.clip(rgb, 0.0, 1.0)
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    src = args[0]
    dst = args[1] if len(args) > 1 else src.replace(".png", "_final.png")
    px, w, h = _load(src)
    _save(process(px), dst, w, h)
    print("ECRIT", dst)


if __name__ == "__main__":
    main()
