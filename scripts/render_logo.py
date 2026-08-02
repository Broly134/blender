"""Rendu studio du logo 3D metallique.

    bpyenv/bin/python scripts/render_logo.py --svg PurePeptide_Symbol.svg \
        --res 1600x1200 --samples 300 --out renders/logo.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

from logo import studio  # noqa: E402
from logo.build import build  # noqa: E402


DEFAULTS = {
    "svg": "PurePeptide_Symbol.svg",
    "out": "renders/logo.png",
    "res": "1400x1400",
    "samples": "300",
    "finish": "polished",
    "depth": "0.052",
    "bevel": "0.0050",
    "exposure": "0.18",
    "view": "hero",
    "roughness": "",
    "alpha": "0",
    "post": "1",
    "dome": "0.44",
}

VIEWS = {
    # (position camera, cible, focale, ouverture)
    "hero": ((-0.78, -3.35, 1.02), (0.0, 0.0, 0.58), 85.0, 10.0),
    "front": ((0.0, -3.60, 0.58), (0.0, 0.0, 0.58), 85.0, 12.0),
    "tilt": ((1.25, -3.05, 1.45), (0.0, 0.0, 0.56), 80.0, 9.0),
    "macro": ((-0.34, -1.25, 0.86), (-0.06, 0.0, 0.68), 100.0, 5.0),
}


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    opts = dict(DEFAULTS)
    for i, a in enumerate(argv):
        if a.startswith("--") and i + 1 < len(argv):
            opts[a[2:]] = argv[i + 1]
    return opts


def main():
    o = parse()
    w, h = (int(v) for v in o["res"].split("x"))
    loc, tgt, focal, fstop = VIEWS[o["view"]]

    studio.reset()
    studio.world()
    objs, extent, lift = build(os.path.abspath(o["svg"]),
                               size=1.0,
                               depth=float(o["depth"]),
                               bevel=float(o["bevel"]),
                               finish=o["finish"],
                               dome=float(o["dome"]),
                               roughness=float(o["roughness"]) if o["roughness"] else None)
    print(f"LOGO {len(objs)} pieces, {extent[0]:.3f} x {extent[1]:.3f}")

    transparent = o["alpha"] == "1"
    if not transparent:
        studio.backdrop()
        studio.reflective_floor()
    studio.lighting()

    # la cible suit la hauteur reelle du logo
    tgt = (tgt[0], tgt[1], lift)
    studio.camera(loc, tgt, focal=focal, fstop=float(fstop))
    studio.render_settings(width=w, height=h, samples=int(o["samples"]),
                           exposure=float(o["exposure"]),
                           output=os.path.abspath(o["out"]),
                           transparent=transparent)

    bpy.ops.render.render(write_still=True)
    print("ECRIT", o["out"])

    if o["post"] != "0" and not transparent:
        from postprocess import process, _load, _save
        px, iw, ih = _load(o["out"])
        _save(process(px, bloom=0.13, vignette=0.22, grain=0.0035),
              os.path.abspath(o["out"]), iw, ih)
        print("POST", o["out"])


main()
