"""Construction du logo 3D a partir du SVG.

Chaque element devient une courbe 2D fermee, extrudee avec un chanfrein arrondi.
Le chanfrein n'est pas cosmetique : c'est lui qui capte les reglettes du studio
et produit le liseré lumineux qui fait lire le metal.
"""

import bpy
import numpy as np

from . import geometry, svgparse
from .metal import logo_material


def grid_faces(rows, cols, close_cols=True):
    faces = []
    for i in range(rows - 1):
        for j in range(cols if close_cols else cols - 1):
            j2 = (j + 1) % cols
            faces.append((i * cols + j, i * cols + j2,
                          (i + 1) * cols + j2, (i + 1) * cols + j))
    return faces


def mesh_from_arrays(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in np.asarray(verts, dtype=float)], [], faces)
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def curve_from_contours(name, contours, depth, bevel, resolution=10):
    """Epaisseur totale = 2*(extrude + bevel) : on retranche donc le chanfrein
    de l'extrusion pour que la piece garde exactement la profondeur demandee."""
    bevel = min(bevel, depth * 0.48)
    cu = bpy.data.curves.new(name, type="CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = max(1e-5, depth * 0.5 - bevel)
    cu.bevel_depth = bevel
    cu.bevel_resolution = resolution
    cu.offset = -bevel                 # garde la silhouette d'origine
    cu.resolution_u = 1
    cu.use_fill_caps = True

    for c in contours:
        sp = cu.splines.new("POLY")
        sp.points.add(len(c) - 1)
        flat = np.zeros((len(c), 4))
        flat[:, :2] = c
        flat[:, 3] = 1.0
        sp.points.foreach_set("co", flat.ravel())
        sp.use_cyclic_u = True

    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_euler = (np.pi / 2.0, 0.0, 0.0)   # le logo se dresse en XZ
    return obj


def _paint_info(shape):
    """(couleur representative, degrade eventuel) d'un element."""
    paint = shape.fill or shape.stroke
    if paint is None:
        return (0.5, 0.5, 0.5), None
    kind, value = paint
    if kind == "color":
        return value, None
    stops = value["stops"]
    mid = stops[len(stops) // 2][1]
    return mid, stops


def build(svg_path, size=1.0, depth=0.052, bevel=0.0050, finish="polished",
          bead_offset=0.014, roughness=None, dome=0.44):
    """Cree tous les objets du logo et renvoie (objets, dimensions)."""
    shapes = svgparse.read(svg_path)

    per_shape = []
    for sh in shapes:
        cs = geometry.contours_of(sh)
        if cs:
            per_shape.append((sh, cs))
    if not per_shape:
        raise ValueError(f"aucune geometrie exploitable dans {svg_path}")

    transform, extent = geometry.normalize(
        [c for _, cs in per_shape for c in cs], size=size)

    objects = []
    for index, (sh, cs) in enumerate(per_shape):
        contours = [transform(c) for c in cs]
        color, gradient = _paint_info(sh)
        name = sh.name or f"piece{index}"

        is_bead = sh.fill is not None and sh.stroke is None
        piece_bevel = bevel
        if is_bead and dome > 0.0:
            # les pastilles pleines deviennent des cabochons bombes : un disque
            # plat ne renvoie qu'un aplat, une calotte accroche la lumiere
            pts = np.concatenate(contours)
            half = float((pts.max(axis=0) - pts.min(axis=0)).min()) * 0.5
            piece_bevel = max(bevel, half * dome)

        obj = curve_from_contours(name, contours, depth, piece_bevel)

        # les billes ressortent legerement du cadre : en 3D, un logo tout plat
        # perd la lecture des superpositions que le SVG donnait par l'ordre
        if is_bead:
            obj.location.y = -bead_offset

        mat = logo_material(f"Mat_{name}", color=color, gradient=gradient,
                            finish=finish, roughness=roughness)
        obj.data.materials.append(mat)
        objects.append(obj)

    # apres normalisation le logo est centre sur l'origine : on le remonte
    # pour le poser sur le sol, avec un leger jeu
    lift = float(extent[1]) * 0.5 + size * 0.025
    for o in objects:
        o.location.z += lift
    return objects, extent, lift
