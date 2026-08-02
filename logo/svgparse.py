"""Lecteur SVG minimal mais suffisant pour un logo de marque.

L'importeur SVG de Blender ne gere ni les contours (`stroke`) ni les degrades :
sur un logo comme celui-ci, ou tout est justement fait de traits epais et de
degrades, il ne restituerait rien. On lit donc le fichier nous-memes et on
renvoie des contours 2D fermes, prets a etre extrudes.

Elements pris en charge : g (transform), circle, ellipse, rect, line, polyline,
polygon, path (M L H V C S Q T A Z, absolus et relatifs), linearGradient.
"""

import math
import re
import xml.etree.ElementTree as ET

import numpy as np


NS = {"svg": "http://www.w3.org/2000/svg"}
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _numbers(text):
    return [float(v) for v in _NUM.findall(text or "")]


# ---------------------------------------------------------------------------
# transformations
# ---------------------------------------------------------------------------

def _identity():
    return np.eye(3)


def parse_transform(text):
    """Compose la liste de transformations SVG en une matrice 3x3."""
    m = _identity()
    if not text:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text):
        v = _numbers(args)
        t = _identity()
        if name == "matrix" and len(v) == 6:
            t = np.array([[v[0], v[2], v[4]],
                          [v[1], v[3], v[5]],
                          [0.0, 0.0, 1.0]])
        elif name == "translate":
            t[0, 2] = v[0]
            t[1, 2] = v[1] if len(v) > 1 else 0.0
        elif name == "scale":
            sx = v[0]
            sy = v[1] if len(v) > 1 else sx
            t[0, 0], t[1, 1] = sx, sy
        elif name == "rotate":
            a = math.radians(v[0])
            c, s = math.cos(a), math.sin(a)
            r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            if len(v) == 3:
                pre, post = _identity(), _identity()
                pre[0, 2], pre[1, 2] = v[1], v[2]
                post[0, 2], post[1, 2] = -v[1], -v[2]
                r = pre @ r @ post
            t = r
        elif name in ("skewX", "skewY"):
            k = math.tan(math.radians(v[0]))
            t[0, 1 if name == "skewX" else 0] = k
        m = m @ t
    return m


def apply(matrix, pts):
    pts = np.asarray(pts, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (homo @ matrix.T)[:, :2]


# ---------------------------------------------------------------------------
# couleurs et degrades
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def parse_color(text):
    """'#RRGGBB' ou '#RGB' -> triplet lineaire."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("#"):
        h = text[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
        return tuple(srgb_to_linear(rgb))
    return None


def parse_gradients(root):
    grads = {}
    for el in root.iter():
        if not el.tag.endswith("linearGradient"):
            continue
        stops = []
        for st in el:
            if not st.tag.endswith("stop"):
                continue
            off = st.get("offset", "0")
            off = float(off[:-1]) / 100.0 if off.endswith("%") else float(off)
            col = parse_color(st.get("stop-color"))
            if col is not None:
                stops.append((off, col))
        if stops:
            def _f(v, default):
                v = el.get(v)
                if v is None:
                    return default
                return float(v[:-1]) / 100.0 if v.endswith("%") else float(v)
            grads[el.get("id")] = {
                "stops": sorted(stops),
                "p0": (_f("x1", 0.0), _f("y1", 0.0)),
                "p1": (_f("x2", 1.0), _f("y2", 0.0)),
            }
    return grads


def _paint(value, grads):
    """Renvoie ('color', rgb) ou ('gradient', def) ou None."""
    if not value or value == "none":
        return None
    value = value.strip()
    m = re.match(r"url\(#([^)]+)\)", value)
    if m:
        g = grads.get(m.group(1))
        return ("gradient", g) if g else None
    c = parse_color(value)
    return ("color", c) if c else None


# ---------------------------------------------------------------------------
# aplatissement des primitives
# ---------------------------------------------------------------------------

def _circle_pts(cx, cy, rx, ry, n=96):
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def _bezier3(p0, p1, p2, p3, n=24):
    t = np.linspace(0.0, 1.0, n + 1)[1:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _bezier2(p0, p1, p2, n=18):
    t = np.linspace(0.0, 1.0, n + 1)[1:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def parse_path(d):
    """Aplatit un attribut `d` en une liste de sous-chemins (points, ferme)."""
    tokens = re.findall(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)", d)
    items = []
    for cmd, num in tokens:
        items.append(cmd if cmd else float(num))

    subpaths = []
    pts = []
    cur = np.zeros(2)
    start = np.zeros(2)
    prev_ctrl = None
    i = 0
    cmd = None
    while i < len(items):
        if isinstance(items[i], str):
            cmd = items[i]
            i += 1
            if cmd in "Zz":
                if len(pts) > 2:
                    subpaths.append((np.array(pts), True))
                pts = []
                cur = start.copy()
                continue
        if cmd is None:
            break

        def take(k):
            nonlocal i
            v = items[i:i + k]
            i += k
            return [float(x) for x in v]

        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = take(2)
            cur = cur + (x, y) if rel else np.array([x, y])
            if len(pts) > 2:
                subpaths.append((np.array(pts), False))
            pts = [cur.copy()]
            start = cur.copy()
            cmd = "l" if rel else "L"
        elif c == "L":
            x, y = take(2)
            cur = cur + (x, y) if rel else np.array([x, y])
            pts.append(cur.copy())
        elif c == "H":
            x = take(1)[0]
            cur = np.array([cur[0] + x if rel else x, cur[1]])
            pts.append(cur.copy())
        elif c == "V":
            y = take(1)[0]
            cur = np.array([cur[0], cur[1] + y if rel else y])
            pts.append(cur.copy())
        elif c in ("C", "S"):
            if c == "C":
                x1, y1, x2, y2, x, y = take(6)
                p1 = cur + (x1, y1) if rel else np.array([x1, y1])
                p2 = cur + (x2, y2) if rel else np.array([x2, y2])
            else:
                x2, y2, x, y = take(4)
                p1 = 2 * cur - prev_ctrl if prev_ctrl is not None else cur
                p2 = cur + (x2, y2) if rel else np.array([x2, y2])
            p3 = cur + (x, y) if rel else np.array([x, y])
            pts.extend(_bezier3(cur, p1, p2, p3))
            prev_ctrl, cur = p2, p3
            continue
        elif c in ("Q", "T"):
            if c == "Q":
                x1, y1, x, y = take(4)
                p1 = cur + (x1, y1) if rel else np.array([x1, y1])
            else:
                x, y = take(2)
                p1 = 2 * cur - prev_ctrl if prev_ctrl is not None else cur
            p2 = cur + (x, y) if rel else np.array([x, y])
            pts.extend(_bezier2(cur, p1, p2))
            prev_ctrl, cur = p1, p2
            continue
        elif c == "A":
            rx, ry, rot, laf, sf, x, y = take(7)
            end = cur + (x, y) if rel else np.array([x, y])
            pts.extend(_arc(cur, end, rx, ry, rot, laf, sf))
            cur = end
        else:
            i += 1
        prev_ctrl = None

    if len(pts) > 2:
        subpaths.append((np.array(pts), False))
    return subpaths


def _arc(p0, p1, rx, ry, rot_deg, large, sweep, n=32):
    """Arc elliptique SVG -> polyligne (conversion endpoint -> centre)."""
    if rx == 0 or ry == 0:
        return [p1]
    phi = math.radians(rot_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx, dy = (p0 - p1) / 2.0
    x1 = cos_p * dx + sin_p * dy
    y1 = -sin_p * dx + cos_p * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = max(rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2, 0.0)
    den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
    coef = math.sqrt(num / den) * (-1 if large == sweep else 1)
    cx1 = coef * rx * y1 / ry
    cy1 = -coef * ry * x1 / rx
    cx = cos_p * cx1 - sin_p * cy1 + (p0[0] + p1[0]) / 2
    cy = sin_p * cx1 + cos_p * cy1 + (p0[1] + p1[1]) / 2

    t0 = math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    t1 = math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx)
    dt = t1 - t0
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    ts = t0 + dt * np.linspace(0.0, 1.0, n + 1)[1:]
    xs = cx + rx * np.cos(ts) * cos_p - ry * np.sin(ts) * sin_p
    ys = cy + rx * np.cos(ts) * sin_p + ry * np.sin(ts) * cos_p
    return list(np.stack([xs, ys], axis=1))


# ---------------------------------------------------------------------------
# lecture complete
# ---------------------------------------------------------------------------

class Shape:
    """Un element du logo, deja aplati et transforme."""

    def __init__(self, name, subpaths, fill=None, stroke=None, width=0.0,
                 closed=True, cap="butt"):
        self.name = name
        self.subpaths = subpaths       # liste de (points Nx2, ferme)
        self.fill = fill               # ('color', rgb) | ('gradient', def) | None
        self.stroke = stroke
        self.width = width
        self.closed = closed
        self.cap = cap


def read(path):
    tree = ET.parse(path)
    root = tree.getroot()
    grads = parse_gradients(root)
    shapes = []

    def walk(node, matrix):
        m = matrix @ parse_transform(node.get("transform"))
        tag = node.tag.split("}")[-1]

        fill = _paint(node.get("fill"), grads)
        stroke = _paint(node.get("stroke"), grads)
        sw = float(node.get("stroke-width") or 0.0)
        cap = node.get("stroke-linecap", "butt")
        name = node.get("id") or tag

        subs = None
        closed = True
        if tag == "circle":
            r = float(node.get("r", 0))
            subs = [(_circle_pts(float(node.get("cx", 0)), float(node.get("cy", 0)),
                                 r, r), True)]
        elif tag == "ellipse":
            subs = [(_circle_pts(float(node.get("cx", 0)), float(node.get("cy", 0)),
                                 float(node.get("rx", 0)), float(node.get("ry", 0))), True)]
        elif tag == "rect":
            x, y = float(node.get("x", 0)), float(node.get("y", 0))
            w, h = float(node.get("width", 0)), float(node.get("height", 0))
            subs = [(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), True)]
        elif tag in ("polyline", "polygon"):
            v = _numbers(node.get("points"))
            subs = [(np.array(v).reshape(-1, 2), tag == "polygon")]
            closed = tag == "polygon"
        elif tag == "line":
            subs = [(np.array([[float(node.get("x1", 0)), float(node.get("y1", 0))],
                               [float(node.get("x2", 0)), float(node.get("y2", 0))]]), False)]
            closed = False
        elif tag == "path":
            subs = parse_path(node.get("d", ""))
            closed = all(c for _, c in subs) if subs else True

        if subs:
            subs = [(apply(m, p), c) for p, c in subs]
            shapes.append(Shape(name, subs, fill, stroke, sw * _scale_of(m),
                                closed, cap))

        for child in node:
            walk(child, m)

    walk(root, _identity())
    return shapes


def _scale_of(matrix):
    """Facteur d'echelle moyen d'une matrice, pour convertir stroke-width."""
    a, b = matrix[0, 0], matrix[1, 0]
    c, d = matrix[0, 1], matrix[1, 1]
    return math.sqrt(abs(a * d - b * c)) or 1.0
