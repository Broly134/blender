"""Construit le personnage complet et lance le rendu.

    bpyenv/bin/python scripts/render.py --res 1024x1536 --samples 320 \
        --out renders/portrait.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

from character import scene as sc      # noqa: E402
from character.build import assemble   # noqa: E402


DEFAULTS = {
    "out": "renders/portrait.png",
    "res": "1024x1536",
    "samples": "300",
    "quality": "1.0",
    "exposure": "0.36",
    "cam": "portrait",
    "hair": "1",
    "save": "",
}

CAMS = {
    # (position, point vise, focale, ouverture)
    "portrait": ((0.0250, 0.6100, 0.0640), (0.0, 0.0450, 0.0840), 32.0, 2.2),
    "wide": ((0.040, 0.720, 0.150), (0.0, 0.030, 0.080), 50.0, 3.5),
    "profile": ((0.520, 0.330, 0.140), (0.0, 0.030, 0.105), 60.0, 4.0),
    "face": ((0.020, 0.380, 0.1120), (0.0, 0.070, 0.1120), 42.0, 2.8),
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
    loc, tgt, focal, fstop = CAMS[o["cam"]]

    sc.reset_scene()
    sc.setup_world(strength=0.34)
    sc.add_sun()
    sc.add_bounce()
    sc.add_ground()
    sc.add_backdrop()

    assemble(quality=float(o["quality"]), with_hair=o["hair"] != "0",
             gaze=(loc[0] * 0.9, loc[1], loc[2]))

    sc.add_camera(location=loc, target=tgt, focal=focal, fstop=float(fstop))
    sc.setup_render(width=w, height=h, samples=int(o["samples"]),
                    exposure=float(o["exposure"]),
                    output=os.path.abspath(o["out"]))

    bpy.ops.render.render(write_still=True)
    print("ECRIT", o["out"])

    if o.get("post", "1") != "0":
        from postprocess import process, _load, _save
        px, w, h = _load(o["out"])
        _save(process(px), os.path.abspath(o["out"]), w, h)
        print("POST", o["out"])

    # la sauvegarde vient en dernier : dans le module bpy, save_as_mainfile
    # remplace le contexte courant et coupe court a tout ce qui suit
    if o["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(o["save"]))
        print("BLEND", o["save"])


main()
