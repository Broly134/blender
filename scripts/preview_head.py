"""Rendu clay rapide de la tete seule, pour valider la sculpture.

    bpyenv/bin/python scripts/preview_head.py --views front,side,three
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402

from character import scene as sc          # noqa: E402
from character import materials as mt      # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    opts = {"out": "renders/clay.png", "samples": 24, "res": "520x780",
            "views": "front", "parts": "head"}
    for i, a in enumerate(argv):
        if a.startswith("--") and i + 1 < len(argv):
            opts[a[2:]] = argv[i + 1]
    w, h = (int(x) for x in opts["res"].split("x"))
    return opts["out"], int(opts["samples"]), w, h, opts["views"], opts["parts"]


def studio_lights():
    """Trois sources neutres : c'est le meilleur eclairage pour juger un volume."""
    specs = [
        ("Key", (-0.75, 1.05, 0.95), 220.0, 0.55, (1.0, 0.98, 0.95)),
        ("Fill", (0.95, 0.75, 0.10), 45.0, 1.2, (0.85, 0.90, 1.0)),
        ("Rim", (0.35, -1.05, 0.75), 160.0, 0.5, (1.0, 1.0, 1.0)),
    ]
    for name, loc, energy, size, color in specs:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.size = size
        data.color = color
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = loc
        d = (-loc[0], -loc[1], 0.11 - loc[2])
        obj.rotation_euler = sc._direction_to_euler(sc._normalize(d))

    world = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.030, 0.033, 0.038, 1.0)
    bg.inputs["Strength"].default_value = 1.0


CAMS = {
    "front": ((0.010, 0.930, 0.118), (0.0, 0.030, 0.104), 50.0),
    "side": ((0.930, 0.010, 0.112), (0.0, 0.005, 0.100), 50.0),
    "three": ((0.630, 0.660, 0.190), (0.0, 0.020, 0.100), 50.0),
    "low": ((0.230, 0.880, -0.020), (0.0, 0.040, 0.100), 50.0),
    "closeup": ((0.020, 0.400, 0.115), (0.0, 0.070, 0.100), 55.0),
}


def main():
    out, samples, w, h, views, parts = parse_args()
    sc.reset_scene()
    studio_lights()

    from character.head import build_head, open_mouth
    head = build_head()
    open_mouth(head)
    clay = mt.clay_material()
    head.data.materials.append(clay)

    if "eyes" in parts or parts == "all":
        from character.eyes import build_eyes, cut_eye_openings
        cut_eye_openings(head)
        build_eyes()
    if "ears" in parts or parts == "all":
        from character.ears import build_ears
        for o in build_ears():
            o.data.materials.append(clay)
    if "teeth" in parts or parts == "all":
        from character.teeth import build_teeth
        build_teeth()

    for name in views.split(","):
        loc, tgt, focal = CAMS[name]
        cam = sc.add_camera(location=loc, target=tgt, focal=focal, fstop=16.0)
        cam.data.dof.use_dof = False
        path = out if len(views.split(",")) == 1 else out.replace(".png", f"_{name}.png")
        sc.setup_render(width=w, height=h, samples=samples, output=os.path.abspath(path))
        bpy.context.scene.view_settings.exposure = 0.0
        bpy.ops.render.render(write_still=True)
        print("ECRIT", path)


main()
