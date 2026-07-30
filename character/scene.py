"""Scene : monde, soleil, camera, sol, reglages de rendu."""

import math

import bpy


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "CENTIMETERS"
    return scene


def setup_world(sun_rotation=(0.0, 0.0, 0.0), strength=0.13):
    """Ciel physique de Nishita : c'est lui qui donne le bleu profond du fond."""
    world = bpy.data.worlds.new("Ciel")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    sky = nt.nodes.new("ShaderNodeTexSky")
    _set_enum(sky, "sky_type", ["MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"])
    for attr, val in (("sun_elevation", math.radians(43.0)),
                      ("sun_rotation", math.radians(-35.0)),
                      ("sun_intensity", 0.0),   # le soleil est une vraie lampe
                      ("altitude", 900.0),
                      ("air_density", 1.0),
                      ("dust_density", 0.45),
                      ("ozone_density", 1.4)):
        if hasattr(sky, attr):
            setattr(sky, attr, val)

    sat = nt.nodes.new("ShaderNodeHueSaturation")
    sat.inputs["Saturation"].default_value = 1.22
    sat.inputs["Value"].default_value = 1.0
    nt.links.new(sky.outputs["Color"], sat.inputs["Color"])

    # le fond vu par la camera est assombri et sature davantage, sans toucher
    # a l'eclairage indirect : c'est ce qui donne le bleu dense d'un ciel d'ete
    deep = nt.nodes.new("ShaderNodeHueSaturation")
    deep.inputs["Saturation"].default_value = 1.70
    deep.inputs["Value"].default_value = 0.62
    nt.links.new(sat.outputs["Color"], deep.inputs["Color"])

    path = nt.nodes.new("ShaderNodeLightPath")
    pick = nt.nodes.new("ShaderNodeMixRGB")
    pick.blend_type = "MIX"
    nt.links.new(path.outputs["Is Camera Ray"], pick.inputs["Fac"])
    nt.links.new(sat.outputs["Color"], pick.inputs["Color1"])
    nt.links.new(deep.outputs["Color"], pick.inputs["Color2"])

    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")

    nt.links.new(pick.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return world


def add_sun(direction=(0.55, 0.60, 0.75), energy=6.0, angle_deg=0.55):
    """Soleil dur de milieu de journee : c'est ce qui cree l'ombre nette du bord du chapeau."""
    data = bpy.data.lights.new("Soleil", type="SUN")
    data.energy = energy
    data.angle = math.radians(angle_deg)
    data.color = (1.0, 0.918, 0.782)
    obj = bpy.data.objects.new("Soleil", data)
    bpy.context.scene.collection.objects.link(obj)

    d = -_normalize(direction)
    obj.rotation_euler = _direction_to_euler(d)
    obj.location = (0, 0, 3)
    return obj


def add_bounce(location=(0.34, 0.72, -0.62), energy=13.0, size=1.4,
               color=(1.0, 0.82, 0.56)):
    """Rebond chaud du sol sec, qui deboucher legerement l'ombre du chapeau."""
    data = bpy.data.lights.new("Rebond", type="AREA")
    data.energy = energy
    data.size = size
    data.color = color
    obj = bpy.data.objects.new("Rebond", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = _direction_to_euler(_normalize(
        (-location[0], -location[1], 0.35 - location[2])))
    return obj


def add_camera(location=(0.028, 0.470, 0.132), target=(0.0, 0.060, 0.108),
               focal=38.0, fstop=2.2):
    data = bpy.data.cameras.new("Camera")
    data.lens = focal
    data.sensor_fit = "VERTICAL"
    data.sensor_height = 24.0
    data.dof.use_dof = True
    data.dof.aperture_fstop = fstop

    obj = bpy.data.objects.new("Camera", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = _look_at(location, target)
    bpy.context.scene.camera = obj

    focus = bpy.data.objects.new("Focus", None)
    focus.location = target
    bpy.context.scene.collection.objects.link(focus)
    data.dof.focus_object = focus
    return obj


def add_ground(size=40.0, z=-1.62):
    from .materials import ground_material
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z))
    obj = bpy.context.active_object
    obj.name = "Sol"
    obj.data.materials.append(ground_material())
    return obj


def add_backdrop():
    """Arriere-plan : talus, buissons secs et touffes d'herbe, tous flous.

    Les elements sont places a hauteur de camera et non au sol, sinon le
    cadrage serre du portrait les fait tomber hors champ.
    """
    import numpy as np
    from .materials import shrub_material, ground_material, dry_grass_material

    rng = np.random.default_rng(11)
    mat_shrub = shrub_material()
    mat_grass = dry_grass_material()
    objs = []

    # talus qui ferme l'horizon derriere les epaules
    for cx, cy, rad, sc, z in ((-5.0, 13.0, 3.2, (3.6, 1.3, 0.86), -2.75),
                               (6.5, 18.0, 4.0, (3.2, 1.2, 0.74), -3.05),
                               (0.5, 30.0, 6.0, (3.0, 1.0, 0.62), -4.10)):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=rad, location=(cx, cy, z))
        o = bpy.context.active_object
        o.scale = sc
        o.data.materials.append(ground_material())
        for p in o.data.polygons:
            p.use_smooth = True
        objs.append(o)

    # buissons secs, echelonnes en profondeur
    for i in range(30):
        ang = rng.uniform(-1.25, 1.25)
        dist = rng.uniform(3.0, 13.0)
        s = rng.uniform(0.22, 0.75) * (0.55 + dist * 0.09)
        x = math.sin(ang) * dist
        y = math.cos(ang) * dist
        z = -1.62 + s * 0.6 + rng.uniform(0.0, 0.35) * dist * 0.10
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=s,
                                              location=(x, y, z))
        o = bpy.context.active_object
        o.scale = (1.0, rng.uniform(0.8, 1.35), rng.uniform(0.45, 0.85))
        o.data.materials.append(mat_shrub)
        for p in o.data.polygons:
            p.use_smooth = True
        objs.append(o)

    # quelques masses proches, tres floues, qui donnent la profondeur
    for cx, cy, cz, rad in ((-1.05, 2.10, -0.62, 0.42), (1.35, 2.60, -0.48, 0.50),
                            (-1.70, 3.40, -0.30, 0.62), (1.90, 3.90, -0.24, 0.58),
                            (-0.90, 5.20, -0.10, 0.75)):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rad,
                                              location=(cx, cy, cz))
        o = bpy.context.active_object
        o.scale = (1.0, 0.9, 0.75)
        o.data.materials.append(mat_shrub)
        for p in o.data.polygons:
            p.use_smooth = True
        objs.append(o)

    # touffes d'herbe seche pres du bord du cadre
    for i in range(22):
        ang = rng.uniform(-1.0, 1.0)
        dist = rng.uniform(1.6, 4.5)
        x = math.sin(ang) * dist
        y = math.cos(ang) * dist
        s = rng.uniform(0.16, 0.34)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=s,
                                              location=(x, y, -1.55 + s))
        o = bpy.context.active_object
        o.scale = (1.0, 1.0, rng.uniform(1.4, 2.6))
        o.data.materials.append(mat_grass)
        for p in o.data.polygons:
            p.use_smooth = True
        objs.append(o)
    return objs


def setup_render(width=1024, height=1536, samples=256, denoise=True,
                 exposure=0.0, output=None):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.012
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.use_denoising = denoise
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except TypeError:
        pass

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.threads_mode = "AUTO"

    _set_enum(scene.view_settings, "view_transform", ["AgX", "Filmic", "Standard"])
    _set_enum(scene.view_settings, "look",
              ["AgX - Medium High Contrast", "Medium High Contrast",
               "Filmic - Medium High Contrast", "None"])
    scene.view_settings.exposure = exposure

    if output:
        scene.render.filepath = output
    return scene


# ---------------------------------------------------------------------------

def _set_enum(owner, attr, candidates):
    """Assigne la premiere valeur d'enum acceptee (les noms changent selon la version)."""
    for c in candidates:
        try:
            setattr(owner, attr, c)
            return c
        except TypeError:
            continue
    return None


def _normalize(v):
    import numpy as np
    a = np.asarray(v, dtype=float)
    return a / max(float((a ** 2).sum()) ** 0.5, 1e-9)


def _direction_to_euler(d):
    """Oriente un objet Blender (-Z local) le long de la direction d."""
    from mathutils import Vector
    return Vector((float(d[0]), float(d[1]), float(d[2]))).to_track_quat("-Z", "Y").to_euler()


def _look_at(location, target):
    from mathutils import Vector
    d = Vector(target) - Vector(location)
    return d.to_track_quat("-Z", "Y").to_euler()
