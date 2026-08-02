"""Studio produit : ce n'est pas le shader qui rend un metal credible, c'est
ce qu'il reflechit.

D'ou un vrai plateau : fond incurve, sol reflechissant dont le flou augmente
avec la distance, une grande boite a lumiere en cle, et surtout des reglettes
etroites et allongees. Ce sont elles qui posent sur les aretes biseautees les
longs eclats verticaux qui font lire « metal poli » plutot que « plastique ».
"""

import math

import bpy
import numpy as np


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _direction_to_euler(d):
    from mathutils import Vector
    return Vector((float(d[0]), float(d[1]), float(d[2]))).to_track_quat("-Z", "Y").to_euler()


def _normalize(v):
    a = np.asarray(v, dtype=float)
    return a / max(float((a ** 2).sum()) ** 0.5, 1e-9)


def _set_enum(owner, attr, candidates):
    if not hasattr(owner, attr):
        return None
    for c in candidates:
        try:
            setattr(owner, attr, c)
            return c
        except TypeError:
            continue
    return None


# ---------------------------------------------------------------------------
# lumieres
# ---------------------------------------------------------------------------

def add_area(name, location, target, size, energy, color=(1, 1, 1),
             shape="RECTANGLE", size_y=None, spread=None):
    data = bpy.data.lights.new(name, type="AREA")
    data.shape = shape
    data.size = size
    if size_y is not None and shape in ("RECTANGLE", "ELLIPSE"):
        data.size_y = size_y
    data.energy = energy
    data.color = color
    if spread is not None and hasattr(data, "spread"):
        data.spread = math.radians(spread)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = _direction_to_euler(
        _normalize(np.asarray(target) - np.asarray(location)))
    return obj


def lighting(scale=1.0, key=105.0, warm=(1.0, 0.97, 0.93), cool=(0.90, 0.95, 1.0)):
    s = scale
    lights = [
        # boite a lumiere principale, haut avant gauche
        add_area("Key", (-2.45 * s, -0.70 * s, 2.75 * s), (0, 0, 0.60 * s),
                 2.0 * s, key * 1.25, warm, size_y=1.5 * s),
        # deboucheur large et doux a droite
        add_area("Fill", (1.95 * s, -1.35 * s, 0.35 * s), (0, 0, 0.02 * s),
                 2.6 * s, key * 0.16, cool, size_y=2.0 * s),
        # reglettes : les longs eclats verticaux sur les aretes
        add_area("StripL", (-1.05 * s, -1.15 * s, 0.55 * s), (0, 0, 0.55 * s),
                 0.085 * s, key * 3.10, (1.0, 1.0, 1.0), size_y=2.6 * s, spread=38),
        add_area("StripR", (1.05 * s, -1.10 * s, 0.60 * s), (0, 0, 0.55 * s),
                 0.075 * s, key * 2.30, (0.96, 0.98, 1.0), size_y=2.4 * s, spread=38),
        # reglette haute : l'eclat horizontal sur le chanfrein superieur
        add_area("StripTop", (0.0, -0.95 * s, 1.35 * s), (0, 0, 0.12 * s),
                 2.4 * s, key * 1.90, (1.0, 0.99, 0.97), size_y=0.085 * s, spread=44),
        # contre-jour, pour detacher la silhouette du fond
        add_area("Rim", (0.85 * s, 1.85 * s, 1.05 * s), (0, 0, 0.55 * s),
                 1.4 * s, key * 0.55, (0.86, 0.93, 1.0), size_y=1.0 * s),
        # accents aux couleurs de la marque : ils reancrent la teinte dans les
        # reflets, la ou un metal poli ne renverrait que du blanc
        add_area("AccentTeal", (-2.30 * s, -0.55 * s, 0.10 * s), (0, 0, 0.50 * s),
                 1.6 * s, key * 0.42, (0.09, 0.85, 0.78), size_y=1.6 * s),
        add_area("AccentBlue", (2.35 * s, 0.15 * s, 1.55 * s), (0, 0, 0.55 * s),
                 1.6 * s, key * 0.40, (0.16, 0.42, 1.0), size_y=1.6 * s),
    ]
    for o in lights:
        o.data.use_shadow = True
    return lights


# ---------------------------------------------------------------------------
# decor
# ---------------------------------------------------------------------------

def backdrop(scale=1.0, y0=1.15, radius=2.0, height=5.0, half_width=8.0,
             segments=48, rows=40):
    """Cyclorama : le sol reflechissant se releve en fond par un raccord
    circulaire, sans arete visible a l'horizon."""
    from .build import mesh_from_arrays, grid_faces

    a = np.linspace(0.0, np.pi / 2.0, rows // 2)
    ys = list(y0 + radius * np.sin(a))
    zs = list(radius * (1.0 - np.cos(a)))
    k = np.linspace(0.0, 1.0, rows - len(ys) + 1)[1:]
    ys += [y0 + radius] * len(k)
    zs += list(radius + (height - radius) * k)

    ys = np.array(ys) * scale
    zs = np.array(zs) * scale
    xs = np.linspace(-half_width, half_width, segments) * scale

    verts = np.stack([np.repeat(xs[None, :], len(ys), axis=0).ravel(),
                      np.repeat(ys[:, None], len(xs), axis=1).ravel(),
                      np.repeat(zs[:, None], len(xs), axis=1).ravel()], axis=1)
    faces = grid_faces(len(ys), len(xs), close_cols=False)
    obj = mesh_from_arrays("Cyclo", verts, faces)
    obj.data.materials.append(backdrop_material())
    return obj


def backdrop_material(name="FondStudio"):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nodes, links = nt.nodes, nt.links

    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-700, 0)
    links.new(coord.outputs["Object"], sep.inputs["Vector"])

    # gradient : un peu plus clair derriere le sujet, sombre sur les bords
    dist = nodes.new("ShaderNodeMath")
    dist.operation = "MULTIPLY"
    dist.location = (-520, -140)
    dist.inputs[1].default_value = 0.085
    links.new(sep.outputs["X"], dist.inputs[0])

    absn = nodes.new("ShaderNodeMath")
    absn.operation = "ABSOLUTE"
    absn.location = (-380, -140)
    links.new(dist.outputs["Value"], absn.inputs[0])

    grad = nodes.new("ShaderNodeMath")
    grad.operation = "MULTIPLY_ADD"
    grad.location = (-240, 0)
    grad.inputs[1].default_value = -0.011
    grad.inputs[2].default_value = 0.014
    links.new(absn.outputs["Value"], grad.inputs[0])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (40, 0)
    links.new(grad.outputs["Value"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.42
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (320, 0)
    links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def reflective_floor(scale=1.0, size=14.0):
    """Sol miroir dont le flou augmente avec la distance : le reflet du logo
    reste net a son pied et se dissout au loin."""
    bpy.ops.mesh.primitive_plane_add(size=size * scale, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "Sol"

    mat = bpy.data.materials.new("SolMiroir")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nodes, links = nt.nodes, nt.links

    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-720, 0)
    links.new(coord.outputs["Object"], sep.inputs["Vector"])

    dist = nodes.new("ShaderNodeMath")
    dist.operation = "ABSOLUTE"
    dist.location = (-560, 0)
    links.new(sep.outputs["Y"], dist.inputs[0])

    rough = nodes.new("ShaderNodeMapRange")
    rough.location = (-380, 0)
    rough.inputs["From Min"].default_value = 0.0
    rough.inputs["From Max"].default_value = 2.4 * scale
    rough.inputs["To Min"].default_value = 0.055
    rough.inputs["To Max"].default_value = 0.48
    links.new(dist.outputs["Value"], rough.inputs["Value"])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (0.0075, 0.0080, 0.0092, 1.0)
    links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.62

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def world(strength=0.022, top=(0.10, 0.12, 0.15), bottom=(0.010, 0.011, 0.013)):
    w = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    nodes, links = nt.nodes, nt.links

    tex = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*bottom, 1.0)
    ramp.color_ramp.elements[1].color = (*top, 1.0)
    links.new(sep.outputs["Z"], ramp.inputs["Fac"])

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nodes.new("ShaderNodeOutputWorld")
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    return w


def camera(location, target, focal=85.0, fstop=8.0):
    data = bpy.data.cameras.new("Camera")
    data.lens = focal
    data.sensor_fit = "AUTO"
    data.dof.use_dof = True
    data.dof.aperture_fstop = fstop
    obj = bpy.data.objects.new("Camera", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = obj

    focus = bpy.data.objects.new("Focus", None)
    focus.location = target
    bpy.context.scene.collection.objects.link(focus)
    data.dof.focus_object = focus
    return obj


def render_settings(width=1600, height=1200, samples=280, exposure=0.0,
                    output=None, transparent=False):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.008
    scene.cycles.max_bounces = 20
    scene.cycles.glossy_bounces = 12       # indispensable entre pieces metalliques
    scene.cycles.diffuse_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except TypeError:
        pass

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"

    _set_enum(scene.view_settings, "view_transform", ["AgX", "Filmic", "Standard"])
    _set_enum(scene.view_settings, "look",
              ["AgX - Medium High Contrast", "Medium High Contrast", "None"])
    scene.view_settings.exposure = exposure
    if output:
        scene.render.filepath = output
    return scene
