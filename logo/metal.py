"""Shaders metalliques pour le logo.

Un metal ne renvoie que ce qu'il reflechit : sa couleur de base agit comme une
reflectance, pas comme un pigment. Une couleur de marque sombre et saturee
donnerait donc un metal terne et gris. On la « remonte » d'abord vers une
reflectance credible (canal max autour de 0,8) en conservant la teinte, ce qui
donne un metal colore lumineux du type anodise.
"""

import colorsys

import bpy
import numpy as np


def _new(name):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat


def set_input(node, names, value):
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def link_input(nt, socket, node, names):
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in node.inputs:
            nt.links.new(socket, node.inputs[n])
            return True
    return False


def metal_tint(rgb, ceiling=0.78, sat=1.08, base=0.30, span=0.52):
    """Couleur de marque -> reflectance metallique credible.

    La clarte d'origine est comprimee dans une plage haute (et non ecrasee sur
    une valeur unique) : c'est ce qui preserve le degrade bleu marine ->
    turquoise des billes, qui autrement se retrouveraient toutes aussi claires.
    """
    r, g, b = (max(0.0, float(c)) for c in rgb)
    h, l, s = colorsys.rgb_to_hls(*np.clip([r ** (1 / 2.2), g ** (1 / 2.2),
                                            b ** (1 / 2.2)], 0, 1))
    s = min(1.0, s * sat)
    l = base + span * l
    out = np.array(colorsys.hls_to_rgb(h, l, s))
    peak = float(out.max()) or 1.0
    if peak > ceiling:
        out = out * (ceiling / peak)
    return tuple(np.clip(out, 0.0, 1.0) ** 2.2)


# ---------------------------------------------------------------------------

def _color_source(nt, color, gradient, location=(-900, 200)):
    """Sortie couleur : uniforme, ou degrade suivant la diagonale de la piece."""
    nodes, links = nt.nodes, nt.links
    if gradient is None:
        rgb = nodes.new("ShaderNodeRGB")
        rgb.location = location
        rgb.outputs[0].default_value = (*color, 1.0)
        return rgb.outputs[0]

    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (location[0] - 500, location[1])
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (location[0] - 320, location[1])
    links.new(coord.outputs["Generated"], sep.inputs["Vector"])

    # le degrade SVG va du coin haut-gauche au coin bas-droit de la piece ;
    # Generated a deja Y vers le haut, d'ou le 1 - y
    inv = nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.location = (location[0] - 160, location[1] - 120)
    inv.inputs[0].default_value = 1.0
    links.new(sep.outputs["Y"], inv.inputs[1])

    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.location = (location[0] - 40, location[1])
    links.new(sep.outputs["X"], add.inputs[0])
    links.new(inv.outputs["Value"], add.inputs[1])

    half = nodes.new("ShaderNodeMath")
    half.operation = "MULTIPLY"
    half.location = (location[0] + 80, location[1])
    half.inputs[1].default_value = 0.5
    links.new(add.outputs["Value"], half.inputs[0])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (location[0] + 220, location[1])
    stops = gradient
    els = ramp.color_ramp.elements
    els[0].position = max(0.0, min(1.0, stops[0][0]))
    els[0].color = (*stops[0][1], 1.0)
    if len(stops) > 1:
        els[1].position = max(0.0, min(1.0, stops[-1][0]))
        els[1].color = (*stops[-1][1], 1.0)
    for pos, col in stops[1:-1]:
        e = els.new(max(0.0, min(1.0, pos)))
        e.color = (*col, 1.0)
    links.new(half.outputs["Value"], ramp.inputs["Fac"])
    return ramp.outputs["Color"]


def logo_material(name, color=(0.5, 0.5, 0.5), gradient=None, finish="polished",
                  roughness=None):
    """Materiau d'une piece du logo.

    finish : 'polished' (metal colore poli), 'brushed' (metal brosse anisotrope),
    'chrome' (metal neutre), 'gold', 'paint' (laque + vernis).
    """
    mat = _new(name)
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    if finish == "chrome":
        color, gradient = (0.85, 0.86, 0.88), None
    elif finish == "gold":
        color, gradient = (0.944, 0.616, 0.234), None

    if finish in ("polished", "brushed", "chrome", "gold"):
        base = metal_tint(color) if finish in ("polished", "brushed") else color
        grad = None
        if gradient is not None and finish in ("polished", "brushed"):
            grad = [(p, metal_tint(c)) for p, c in gradient]
    else:
        base, grad = color, gradient

    src = _color_source(nt, base, grad)

    # micro-relief : sans lui, un metal poli est trop parfait et se lit en CG
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, -420)
    micro = nodes.new("ShaderNodeTexNoise")
    micro.location = (-700, -420)
    micro.inputs["Scale"].default_value = 260.0
    micro.inputs["Detail"].default_value = 6.0
    micro.inputs["Roughness"].default_value = 0.6
    links.new(coord.outputs["Object"], micro.inputs["Vector"])

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-460, -420)
    bump.inputs["Strength"].default_value = 0.045
    bump.inputs["Distance"].default_value = 0.0008
    links.new(micro.outputs["Fac"], bump.inputs["Height"])

    # variation de rugosite : les reflets ne sont jamais uniformes
    rough_tex = nodes.new("ShaderNodeTexNoise")
    rough_tex.location = (-700, -640)
    rough_tex.inputs["Scale"].default_value = 14.0
    rough_tex.inputs["Detail"].default_value = 4.0
    links.new(coord.outputs["Object"], rough_tex.inputs["Vector"])

    defaults = {"polished": 0.095, "brushed": 0.250, "chrome": 0.045,
                "gold": 0.130, "paint": 0.140}
    r0 = roughness if roughness is not None else defaults.get(finish, 0.10)

    rmap = nodes.new("ShaderNodeMapRange")
    rmap.location = (-460, -640)
    rmap.inputs["From Min"].default_value = 0.3
    rmap.inputs["From Max"].default_value = 0.7
    rmap.inputs["To Min"].default_value = max(0.01, r0 * 0.72)
    rmap.inputs["To Max"].default_value = r0 * 1.32
    links.new(rough_tex.outputs["Fac"], rmap.inputs["Value"])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(src, bsdf.inputs["Base Color"])
    link_input(nt, rmap.outputs["Result"], bsdf, "Roughness")
    link_input(nt, bump.outputs["Normal"], bsdf, "Normal")

    if finish == "paint":
        set_input(bsdf, ["Metallic"], 0.25)
        set_input(bsdf, ["Coat Weight", "Clearcoat"], 1.0)
        set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.035)
        set_input(bsdf, ["Specular IOR Level", "Specular"], 0.55)
    else:
        set_input(bsdf, ["Metallic"], 1.0)
        set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.22)
        set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.045)

    if finish == "brushed":
        set_input(bsdf, ["Anisotropic"], 0.82)
        set_input(bsdf, ["Anisotropic Rotation"], 0.0)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat
