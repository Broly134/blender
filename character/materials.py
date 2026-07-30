"""Shaders : peau (SSS + pores procéduraux), feutre, cuir, verre, tissu, blé.

Les variations regionales de la peau (levres plus rouges, cerne, ombre de
barbe, zones grasses du front et du nez) ne sont pas peintes a la main : elles
sont ecrites dans des attributs de sommets calcules analytiquement dans
`skin_attributes.py`, puis lues par le shader.
"""

import bpy


# ---------------------------------------------------------------------------
# utilitaires
# ---------------------------------------------------------------------------

def set_input(node, names, value):
    """Assigne une entree en essayant plusieurs noms (les sockets du Principled
    ont ete renommes entre Blender 3.x, 4.x et 5.x)."""
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def link_input(nt, out_socket, node, names):
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in node.inputs:
            nt.links.new(out_socket, node.inputs[n])
            return True
    return False


def new_material(name, use_nodes=True):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = use_nodes
    if use_nodes:
        mat.node_tree.nodes.clear()
    return mat


def _principled(nt, location=(0, 0)):
    n = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n.location = location
    return n


def _output(nt, shader, location=(400, 0)):
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = location
    nt.links.new(shader.outputs[0], out.inputs["Surface"])
    return out


# ---------------------------------------------------------------------------
# peau
# ---------------------------------------------------------------------------

def skin_material(name="Peau"):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-1600, 0)

    # --- attributs peints proceduralement -----------------------------------
    tint = nodes.new("ShaderNodeVertexColor")
    tint.layer_name = "skin_tint"
    tint.location = (-1600, -400)

    oil = nodes.new("ShaderNodeAttribute")
    oil.attribute_name = "skin_oil"
    oil.location = (-1600, -560)

    sssw = nodes.new("ShaderNodeAttribute")
    sssw.attribute_name = "skin_sss"
    sssw.location = (-1600, -700)

    # --- couleur de base ----------------------------------------------------
    base = nodes.new("ShaderNodeRGB")
    base.outputs[0].default_value = (0.0790, 0.0268, 0.0150, 1.0)
    base.location = (-1300, 200)

    # taches de melanine basse frequence
    blotch = nodes.new("ShaderNodeTexNoise")
    blotch.location = (-1300, -60)
    blotch.inputs["Scale"].default_value = 26.0
    blotch.inputs["Detail"].default_value = 5.0
    blotch.inputs["Roughness"].default_value = 0.62
    links.new(coord.outputs["Object"], blotch.inputs["Vector"])

    blotch_ramp = nodes.new("ShaderNodeValToRGB")
    blotch_ramp.location = (-1080, -60)
    blotch_ramp.color_ramp.interpolation = "EASE"
    blotch_ramp.color_ramp.elements[0].position = 0.36
    blotch_ramp.color_ramp.elements[0].color = (0.905, 0.880, 0.865, 1.0)
    blotch_ramp.color_ramp.elements[1].position = 0.66
    blotch_ramp.color_ramp.elements[1].color = (1.075, 1.055, 1.040, 1.0)
    links.new(blotch.outputs["Fac"], blotch_ramp.inputs["Fac"])

    mix_blotch = nodes.new("ShaderNodeMixRGB")
    mix_blotch.blend_type = "MULTIPLY"
    mix_blotch.inputs["Fac"].default_value = 0.55
    mix_blotch.location = (-860, 120)
    links.new(base.outputs[0], mix_blotch.inputs["Color1"])
    links.new(blotch_ramp.outputs["Color"], mix_blotch.inputs["Color2"])

    mix_tint = nodes.new("ShaderNodeMixRGB")
    mix_tint.blend_type = "MULTIPLY"
    mix_tint.inputs["Fac"].default_value = 1.0
    mix_tint.location = (-640, 120)
    links.new(mix_blotch.outputs["Color"], mix_tint.inputs["Color1"])
    links.new(tint.outputs["Color"], mix_tint.inputs["Color2"])

    # --- rugosite -----------------------------------------------------------
    rough_noise = nodes.new("ShaderNodeTexNoise")
    rough_noise.location = (-1300, -300)
    rough_noise.inputs["Scale"].default_value = 46.0
    rough_noise.inputs["Detail"].default_value = 6.0
    links.new(coord.outputs["Object"], rough_noise.inputs["Vector"])

    rough_map = nodes.new("ShaderNodeMapRange")
    rough_map.location = (-1080, -300)
    rough_map.inputs["From Min"].default_value = 0.25
    rough_map.inputs["From Max"].default_value = 0.75
    rough_map.inputs["To Min"].default_value = 0.62
    rough_map.inputs["To Max"].default_value = 0.44
    links.new(rough_noise.outputs["Fac"], rough_map.inputs["Value"])

    # les zones grasses (front, nez, pommettes) deviennent plus lisses
    rough_oil = nodes.new("ShaderNodeMath")
    rough_oil.operation = "MULTIPLY_ADD"
    rough_oil.location = (-860, -300)
    rough_oil.inputs[1].default_value = -0.135
    links.new(oil.outputs["Fac"], rough_oil.inputs[0])
    links.new(rough_map.outputs["Result"], rough_oil.inputs[2])

    # --- micro-relief : pores + grain --------------------------------------
    pores = nodes.new("ShaderNodeTexVoronoi")
    pores.location = (-1300, -820)
    pores.feature = "F1"
    pores.inputs["Scale"].default_value = 1150.0
    links.new(coord.outputs["Object"], pores.inputs["Vector"])

    pores_ramp = nodes.new("ShaderNodeMapRange")
    pores_ramp.location = (-1080, -820)
    pores_ramp.inputs["From Min"].default_value = 0.0
    pores_ramp.inputs["From Max"].default_value = 0.55
    pores_ramp.inputs["To Min"].default_value = 0.0
    pores_ramp.inputs["To Max"].default_value = 1.0
    links.new(pores.outputs["Distance"], pores_ramp.inputs["Value"])

    grain = nodes.new("ShaderNodeTexNoise")
    grain.location = (-1300, -1050)
    grain.inputs["Scale"].default_value = 320.0
    grain.inputs["Detail"].default_value = 8.0
    grain.inputs["Roughness"].default_value = 0.72
    links.new(coord.outputs["Object"], grain.inputs["Vector"])

    micro = nodes.new("ShaderNodeTexNoise")
    micro.location = (-1300, -1260)
    micro.inputs["Scale"].default_value = 1900.0
    micro.inputs["Detail"].default_value = 4.0
    links.new(coord.outputs["Object"], micro.inputs["Vector"])

    meso = nodes.new("ShaderNodeTexNoise")
    meso.location = (-1300, -700)
    meso.inputs["Scale"].default_value = 85.0
    meso.inputs["Detail"].default_value = 7.0
    meso.inputs["Roughness"].default_value = 0.65
    links.new(coord.outputs["Object"], meso.inputs["Vector"])

    bump0 = nodes.new("ShaderNodeBump")
    bump0.location = (-820, -900)
    bump0.inputs["Strength"].default_value = 0.34
    bump0.inputs["Distance"].default_value = 0.0042
    links.new(meso.outputs["Fac"], bump0.inputs["Height"])

    bump1 = nodes.new("ShaderNodeBump")
    bump1.location = (-620, -900)
    bump1.inputs["Strength"].default_value = 0.34
    bump1.inputs["Distance"].default_value = 0.0018
    links.new(grain.outputs["Fac"], bump1.inputs["Height"])
    links.new(bump0.outputs["Normal"], bump1.inputs["Normal"])

    bump2 = nodes.new("ShaderNodeBump")
    bump2.location = (-420, -900)
    bump2.inputs["Strength"].default_value = 0.30
    bump2.inputs["Distance"].default_value = 0.0006
    links.new(pores_ramp.outputs["Result"], bump2.inputs["Height"])
    links.new(bump1.outputs["Normal"], bump2.inputs["Normal"])

    bump3 = nodes.new("ShaderNodeBump")
    bump3.location = (-220, -900)
    bump3.inputs["Strength"].default_value = 0.12
    bump3.inputs["Distance"].default_value = 0.0002
    links.new(micro.outputs["Fac"], bump3.inputs["Height"])
    links.new(bump2.outputs["Normal"], bump3.inputs["Normal"])

    # --- BSDF ---------------------------------------------------------------
    bsdf = _principled(nt, (60, 0))
    links.new(mix_tint.outputs["Color"], bsdf.inputs["Base Color"])
    link_input(nt, rough_oil.outputs["Value"], bsdf, "Roughness")
    link_input(nt, bump3.outputs["Normal"], bsdf, "Normal")
    set_input(bsdf, ["IOR"], 1.42)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.42)
    set_input(bsdf, ["Specular Tint"], (1.0, 0.90, 0.80, 1.0))

    # diffusion sous la peau : discrete sur une peau tres pigmentee,
    # mais indispensable sur les oreilles, les ailes du nez et les levres
    if not link_input(nt, sssw.outputs["Fac"], bsdf, ["Subsurface Weight", "Subsurface"]):
        pass
    set_input(bsdf, ["Subsurface Radius"], (1.0, 0.34, 0.20))
    set_input(bsdf, ["Subsurface Scale"], 0.0080)
    set_input(bsdf, ["Subsurface Anisotropy"], 0.80)

    # fin vernis : la sueur / le sebum en plein soleil
    set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.055)
    set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.24)

    _output(nt, bsdf, (400, 0))
    return mat


# ---------------------------------------------------------------------------
# yeux
# ---------------------------------------------------------------------------

def eye_material(name="Oeil"):
    """Sclere + iris + pupille en un seul shader, pilote par la coordonnee Z locale."""
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-1400, 0)
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-1200, 0)
    links.new(coord.outputs["Object"], sep.inputs["Vector"])

    # Y local = axe du regard ; on normalise par l'apex corneen (13,8 mm)
    norm = nodes.new("ShaderNodeMapRange")
    norm.location = (-1000, 0)
    norm.inputs["From Min"].default_value = 0.0
    norm.inputs["From Max"].default_value = 0.0138
    links.new(sep.outputs["Y"], norm.inputs["Value"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-800, 0)
    ramp.color_ramp.interpolation = "LINEAR"
    e = ramp.color_ramp.elements
    e[0].position = 0.000
    e[0].color = (0.395, 0.335, 0.280, 1.0)   # sclere, chaude et jamais blanche
    e[1].position = 0.762
    e[1].color = (0.440, 0.375, 0.312, 1.0)
    for pos, col in (
        (0.7760, (0.0210, 0.0100, 0.0052, 1.0)),   # anneau limbique
        (0.7930, (0.1900, 0.0900, 0.0380, 1.0)),   # iris brun
        (0.8450, (0.2600, 0.1320, 0.0560, 1.0)),
        (0.8690, (0.1300, 0.0610, 0.0260, 1.0)),
        (0.8745, (0.0016, 0.0016, 0.0016, 1.0)),   # pupille
        (1.0000, (0.0016, 0.0016, 0.0016, 1.0)),
    ):
        el = ramp.color_ramp.elements.new(pos)
        el.color = col
    links.new(norm.outputs["Result"], ramp.inputs["Fac"])

    # fibres de l'iris
    fibers = nodes.new("ShaderNodeTexNoise")
    fibers.location = (-800, -320)
    fibers.inputs["Scale"].default_value = 220.0
    fibers.inputs["Detail"].default_value = 6.0
    links.new(coord.outputs["Object"], fibers.inputs["Vector"])

    mixf = nodes.new("ShaderNodeMixRGB")
    mixf.blend_type = "OVERLAY"
    mixf.inputs["Fac"].default_value = 0.28
    mixf.location = (-520, 0)
    links.new(ramp.outputs["Color"], mixf.inputs["Color1"])
    links.new(fibers.outputs["Color"], mixf.inputs["Color2"])

    # veinules de la sclere
    veins = nodes.new("ShaderNodeTexNoise")
    veins.location = (-800, -540)
    veins.inputs["Scale"].default_value = 55.0
    veins.inputs["Detail"].default_value = 8.0
    veins.inputs["Roughness"].default_value = 0.8

    bsdf = _principled(nt, (-200, 0))
    links.new(mixf.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ["Roughness"], 0.055)
    set_input(bsdf, ["IOR"], 1.376)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.6)
    set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.14)
    set_input(bsdf, ["Subsurface Radius"], (1.0, 0.55, 0.42))
    set_input(bsdf, ["Subsurface Scale"], 0.004)
    set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.0)
    set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.05)

    _output(nt, bsdf, (120, 0))
    return mat


def teeth_material(name="Dents"):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 90.0
    noise.location = (-600, 0)
    links.new(coord.outputs["Object"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-380, 0)
    ramp.color_ramp.elements[0].color = (0.60, 0.545, 0.470, 1.0)
    ramp.color_ramp.elements[1].color = (0.80, 0.760, 0.690, 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

    bsdf = _principled(nt, (-100, 0))
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ["Roughness"], 0.18)
    set_input(bsdf, ["IOR"], 1.63)
    set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.22)
    set_input(bsdf, ["Subsurface Radius"], (1.0, 0.75, 0.62))
    set_input(bsdf, ["Subsurface Scale"], 0.003)
    set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.6)
    set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.09)
    _output(nt, bsdf)
    return mat


def mouth_material(name="Bouche"):
    mat = new_material(name)
    nt = mat.node_tree
    bsdf = _principled(nt, (0, 0))
    set_input(bsdf, ["Base Color"], (0.055, 0.011, 0.009, 1.0))
    set_input(bsdf, ["Roughness"], 0.24)
    set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.25)
    set_input(bsdf, ["Subsurface Radius"], (1.0, 0.25, 0.2))
    set_input(bsdf, ["Subsurface Scale"], 0.006)
    _output(nt, bsdf)
    return mat


# ---------------------------------------------------------------------------
# poils
# ---------------------------------------------------------------------------

def hair_material(name="Poil", color=(0.0115, 0.0075, 0.0052),
                  roughness=0.32, radial=0.42, grey=0.0):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    hair = nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.location = (0, 0)
    hair.parametrization = "COLOR"
    set_input(hair, ["Color"], (*color, 1.0))
    set_input(hair, ["Roughness"], roughness)
    set_input(hair, ["Radial Roughness"], radial)
    set_input(hair, ["Coat"], 0.08)
    set_input(hair, ["IOR"], 1.55)
    set_input(hair, ["Random Color"], 0.10)
    set_input(hair, ["Random Roughness"], 0.22)

    if grey > 0.0:
        info = nodes.new("ShaderNodeObjectInfo")
        info.location = (-700, -200)
        rnd = nodes.new("ShaderNodeTexNoise")
        rnd.location = (-700, 0)
        rnd.inputs["Scale"].default_value = 300.0
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (-460, 0)
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].color = (*color, 1.0)
        ramp.color_ramp.elements[1].position = 1.0 - grey
        ramp.color_ramp.elements[1].color = (0.30, 0.285, 0.27, 1.0)
        links.new(rnd.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], hair.inputs["Color"])

    _output(nt, hair, (300, 0))
    return mat


# ---------------------------------------------------------------------------
# accessoires
# ---------------------------------------------------------------------------

def felt_material(name="Feutre", color=(0.0655, 0.0262, 0.0104)):
    """Feutre de laine : diffus, duveteux, avec un fin duvet en sheen."""
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-1000, 0)

    fuzz = nodes.new("ShaderNodeTexNoise")
    fuzz.location = (-780, 0)
    fuzz.inputs["Scale"].default_value = 700.0
    fuzz.inputs["Detail"].default_value = 6.0
    fuzz.inputs["Roughness"].default_value = 0.75
    links.new(coord.outputs["Object"], fuzz.inputs["Vector"])

    weave = nodes.new("ShaderNodeTexNoise")
    weave.location = (-780, -240)
    weave.inputs["Scale"].default_value = 90.0
    weave.inputs["Detail"].default_value = 4.0
    links.new(coord.outputs["Object"], weave.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-540, -240)
    ramp.color_ramp.elements[0].color = (0.78, 0.75, 0.72, 1.0)
    ramp.color_ramp.elements[1].color = (1.14, 1.10, 1.05, 1.0)
    links.new(weave.outputs["Fac"], ramp.inputs["Fac"])

    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (*color, 1.0)
    rgb.location = (-540, 120)

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.8
    mix.location = (-320, 60)
    links.new(rgb.outputs[0], mix.inputs["Color1"])
    links.new(ramp.outputs["Color"], mix.inputs["Color2"])

    bmp = nodes.new("ShaderNodeBump")
    bmp.location = (-320, -300)
    bmp.inputs["Strength"].default_value = 0.55
    bmp.inputs["Distance"].default_value = 0.0016
    links.new(fuzz.outputs["Fac"], bmp.inputs["Height"])

    bsdf = _principled(nt, (-60, 0))
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    link_input(nt, bmp.outputs["Normal"], bsdf, "Normal")
    set_input(bsdf, ["Roughness"], 0.90)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.10)
    set_input(bsdf, ["Sheen Weight", "Sheen"], 0.12)
    set_input(bsdf, ["Sheen Roughness"], 0.55)
    set_input(bsdf, ["Sheen Tint"], (0.34, 0.25, 0.19, 1.0))
    _output(nt, bsdf, (220, 0))
    return mat


def leather_material(name="Cuir", color=(0.0455, 0.0142, 0.0072)):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)

    grain = nodes.new("ShaderNodeTexVoronoi")
    grain.location = (-680, -200)
    grain.inputs["Scale"].default_value = 480.0
    grain.feature = "F1"
    links.new(coord.outputs["Object"], grain.inputs["Vector"])

    fine = nodes.new("ShaderNodeTexNoise")
    fine.location = (-680, -430)
    fine.inputs["Scale"].default_value = 260.0
    fine.inputs["Detail"].default_value = 6.0
    links.new(coord.outputs["Object"], fine.inputs["Vector"])

    b1 = nodes.new("ShaderNodeBump")
    b1.location = (-420, -300)
    b1.inputs["Strength"].default_value = 0.42
    b1.inputs["Distance"].default_value = 0.0007
    links.new(grain.outputs["Distance"], b1.inputs["Height"])

    b2 = nodes.new("ShaderNodeBump")
    b2.location = (-220, -300)
    b2.inputs["Strength"].default_value = 0.18
    b2.inputs["Distance"].default_value = 0.0003
    links.new(fine.outputs["Fac"], b2.inputs["Height"])
    links.new(b1.outputs["Normal"], b2.inputs["Normal"])

    rough = nodes.new("ShaderNodeMapRange")
    rough.location = (-420, 200)
    rough.inputs["To Min"].default_value = 0.46
    rough.inputs["To Max"].default_value = 0.68
    links.new(fine.outputs["Fac"], rough.inputs["Value"])

    bsdf = _principled(nt, (40, 0))
    set_input(bsdf, ["Base Color"], (*color, 1.0))
    link_input(nt, rough.outputs["Result"], bsdf, "Roughness")
    link_input(nt, b2.outputs["Normal"], bsdf, "Normal")
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.16)
    _output(nt, bsdf, (300, 0))
    return mat


def metal_material(name="Metal", color=(0.62, 0.60, 0.57), roughness=0.22):
    mat = new_material(name)
    nt = mat.node_tree
    bsdf = _principled(nt)
    set_input(bsdf, ["Base Color"], (*color, 1.0))
    set_input(bsdf, ["Metallic"], 1.0)
    set_input(bsdf, ["Roughness"], roughness)
    _output(nt, bsdf)
    return mat


def plastic_material(name="Acetate", color=(0.0031, 0.0031, 0.0036),
                     roughness=0.26):
    """Acetate noir des lunettes : profond, avec un vernis net."""
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-700, -200)
    micro = nodes.new("ShaderNodeTexNoise")
    micro.location = (-500, -200)
    micro.inputs["Scale"].default_value = 900.0
    links.new(coord.outputs["Object"], micro.inputs["Vector"])
    bmp = nodes.new("ShaderNodeBump")
    bmp.location = (-280, -200)
    bmp.inputs["Strength"].default_value = 0.06
    bmp.inputs["Distance"].default_value = 0.0002
    links.new(micro.outputs["Fac"], bmp.inputs["Height"])

    bsdf = _principled(nt)
    set_input(bsdf, ["Base Color"], (*color, 1.0))
    set_input(bsdf, ["Roughness"], roughness)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.30)
    set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.12)
    set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.10)
    link_input(nt, bmp.outputs["Normal"], bsdf, "Normal")
    _output(nt, bsdf)
    return mat


def lens_material(name="Verre"):
    """Verre correcteur traite antireflet : on doit voir les yeux derriere,
    avec seulement un voile de ciel aux angles rasants."""
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    fres = nodes.new("ShaderNodeFresnel")
    fres.location = (-600, 100)
    fres.inputs["IOR"].default_value = 1.50

    damp = nodes.new("ShaderNodeMath")
    damp.operation = "MULTIPLY"
    damp.location = (-400, 100)
    damp.inputs[1].default_value = 0.085
    links.new(fres.outputs["Fac"], damp.inputs[0])

    boost = nodes.new("ShaderNodeMath")
    boost.operation = "ADD"
    boost.location = (-240, 100)
    boost.inputs[1].default_value = 0.010
    links.new(damp.outputs["Value"], boost.inputs[0])

    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (-240, -60)
    transp.inputs["Color"].default_value = (0.94, 0.955, 0.97, 1.0)

    gloss = nodes.new("ShaderNodeBsdfGlossy")
    gloss.location = (-240, -220)
    gloss.inputs["Roughness"].default_value = 0.015
    gloss.inputs["Color"].default_value = (0.88, 0.92, 1.0, 1.0)

    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (0, 0)
    links.new(boost.outputs["Value"], mix.inputs["Fac"])
    links.new(transp.outputs[0], mix.inputs[1])
    links.new(gloss.outputs[0], mix.inputs[2])

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (240, 0)
    links.new(mix.outputs[0], out.inputs["Surface"])
    mat.use_backface_culling = False
    return mat


def fabric_material(name="Tissu", color=(0.0125, 0.0125, 0.0132),
                    sheen=0.20, scale=420.0, roughness=0.86):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)

    weave = nodes.new("ShaderNodeTexNoise")
    weave.location = (-680, -220)
    weave.inputs["Scale"].default_value = scale
    weave.inputs["Detail"].default_value = 7.0
    weave.inputs["Roughness"].default_value = 0.7
    links.new(coord.outputs["Object"], weave.inputs["Vector"])

    wrinkle = nodes.new("ShaderNodeTexNoise")
    wrinkle.location = (-680, -450)
    wrinkle.inputs["Scale"].default_value = 22.0
    wrinkle.inputs["Detail"].default_value = 4.0
    links.new(coord.outputs["Object"], wrinkle.inputs["Vector"])

    b1 = nodes.new("ShaderNodeBump")
    b1.location = (-420, -350)
    b1.inputs["Strength"].default_value = 0.45
    b1.inputs["Distance"].default_value = 0.0016
    links.new(wrinkle.outputs["Fac"], b1.inputs["Height"])

    b2 = nodes.new("ShaderNodeBump")
    b2.location = (-220, -350)
    b2.inputs["Strength"].default_value = 0.30
    b2.inputs["Distance"].default_value = 0.0005
    links.new(weave.outputs["Fac"], b2.inputs["Height"])
    links.new(b1.outputs["Normal"], b2.inputs["Normal"])

    bsdf = _principled(nt, (40, 0))
    set_input(bsdf, ["Base Color"], (*color, 1.0))
    set_input(bsdf, ["Roughness"], roughness)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.12)
    set_input(bsdf, ["Sheen Weight", "Sheen"], sheen)
    set_input(bsdf, ["Sheen Roughness"], 0.35)
    link_input(nt, b2.outputs["Normal"], bsdf, "Normal")
    _output(nt, bsdf, (300, 0))
    return mat


def straw_material(name="Ble"):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-800, 0)
    n = nodes.new("ShaderNodeTexNoise")
    n.location = (-600, 0)
    n.inputs["Scale"].default_value = 120.0
    n.inputs["Detail"].default_value = 5.0
    links.new(coord.outputs["Object"], n.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-380, 0)
    ramp.color_ramp.elements[0].color = (0.290, 0.180, 0.055, 1.0)
    ramp.color_ramp.elements[1].color = (0.640, 0.470, 0.180, 1.0)
    links.new(n.outputs["Fac"], ramp.inputs["Fac"])

    bsdf = _principled(nt, (-100, 0))
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ["Roughness"], 0.46)
    set_input(bsdf, ["Sheen Weight", "Sheen"], 0.5)
    set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.30)
    set_input(bsdf, ["Subsurface Radius"], (1.0, 0.85, 0.45))
    set_input(bsdf, ["Subsurface Scale"], 0.004)
    _output(nt, bsdf)
    return mat


# ---------------------------------------------------------------------------
# decor
# ---------------------------------------------------------------------------

def ground_material(name="Terre"):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)
    n1 = nodes.new("ShaderNodeTexNoise")
    n1.location = (-700, 0)
    n1.inputs["Scale"].default_value = 3.0
    n1.inputs["Detail"].default_value = 10.0
    links.new(coord.outputs["Object"], n1.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-460, 0)
    ramp.color_ramp.elements[0].color = (0.088, 0.048, 0.017, 1.0)
    ramp.color_ramp.elements[1].color = (0.250, 0.152, 0.055, 1.0)
    links.new(n1.outputs["Fac"], ramp.inputs["Fac"])
    bsdf = _principled(nt, (-160, 0))
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ["Roughness"], 0.96)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.04)
    _output(nt, bsdf)
    return mat


def shrub_material(name="Buisson"):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    n = nodes.new("ShaderNodeTexNoise")
    n.location = (-600, 0)
    n.inputs["Scale"].default_value = 26.0
    n.inputs["Detail"].default_value = 8.0
    links.new(coord.outputs["Object"], n.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-380, 0)
    ramp.color_ramp.elements[0].color = (0.055, 0.062, 0.026, 1.0)
    ramp.color_ramp.elements[1].color = (0.215, 0.230, 0.105, 1.0)
    links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    bsdf = _principled(nt, (-120, 0))
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ["Roughness"], 0.72)
    _output(nt, bsdf)
    return mat


def dry_grass_material(name="HerbeSeche"):
    mat = new_material(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    coord = nodes.new("ShaderNodeTexCoord")
    n = nodes.new("ShaderNodeTexNoise")
    n.location = (-600, 0)
    n.inputs["Scale"].default_value = 42.0
    n.inputs["Detail"].default_value = 8.0
    links.new(coord.outputs["Object"], n.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-380, 0)
    ramp.color_ramp.elements[0].color = (0.170, 0.132, 0.052, 1.0)
    ramp.color_ramp.elements[1].color = (0.480, 0.398, 0.170, 1.0)
    links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    bsdf = _principled(nt, (-120, 0))
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ["Roughness"], 0.62)
    set_input(bsdf, ["Sheen Weight", "Sheen"], 0.45)
    _output(nt, bsdf)
    return mat


def clay_material(name="Clay", color=(0.35, 0.31, 0.29)):
    mat = new_material(name)
    nt = mat.node_tree
    bsdf = _principled(nt)
    set_input(bsdf, ["Base Color"], (*color, 1.0))
    set_input(bsdf, ["Roughness"], 0.55)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.35)
    _output(nt, bsdf)
    return mat
