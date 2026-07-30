"""Helpers Blender : creation d'objets, grilles lofees, groupes de vertices."""

import bmesh
import bpy
import numpy as np


def link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def object_from_arrays(name, verts, faces, collection=None, smooth=True):
    """Cree un objet maillage a partir d'un tableau numpy (N,3) et d'une liste de faces."""
    verts = np.asarray(verts, dtype=np.float32)
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())

    loops = []
    loop_starts = []
    loop_totals = []
    cursor = 0
    for f in faces:
        loop_starts.append(cursor)
        loop_totals.append(len(f))
        loops.extend(f)
        cursor += len(f)

    mesh.loops.add(len(loops))
    mesh.loops.foreach_set("vertex_index", loops)
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", loop_starts)
    mesh.polygons.foreach_set("loop_total", loop_totals)
    if smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(faces))

    mesh.update(calc_edges=True)
    mesh.validate(verbose=False)

    obj = bpy.data.objects.new(name, mesh)
    return link(obj, collection)


def grid_faces(rows, cols, close_cols=True, flip=False):
    """Faces quad d'une grille rows x cols, eventuellement refermee en colonnes."""
    faces = []
    for i in range(rows - 1):
        for j in range(cols if close_cols else cols - 1):
            j2 = (j + 1) % cols
            a = i * cols + j
            b = i * cols + j2
            c = (i + 1) * cols + j2
            d = (i + 1) * cols + j
            faces.append((d, c, b, a) if flip else (a, b, c, d))
    return faces


def cap_ring(start_index, count, flip=False):
    """Un n-gon qui bouche un anneau de `count` sommets."""
    ring = list(range(start_index, start_index + count))
    return tuple(reversed(ring)) if flip else tuple(ring)


def revolve(profile_xy, segments, close_profile=False, flip=False):
    """Revolution d'un profil (r, z) autour de l'axe Z."""
    prof = np.asarray(profile_xy, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = np.empty((len(prof) * segments, 3))
    for i, (r, z) in enumerate(prof):
        verts[i * segments:(i + 1) * segments, 0] = r * np.cos(theta)
        verts[i * segments:(i + 1) * segments, 1] = r * np.sin(theta)
        verts[i * segments:(i + 1) * segments, 2] = z
    faces = grid_faces(len(prof), segments, close_cols=True, flip=flip)
    return verts, faces


def uv_sphere_arrays(radius=1.0, rings=32, segments=48, center=(0, 0, 0)):
    phi = np.linspace(0.0, np.pi, rings)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    P, T = np.meshgrid(phi, theta, indexing="ij")
    x = radius * np.sin(P) * np.cos(T)
    y = radius * np.sin(P) * np.sin(T)
    z = radius * np.cos(P)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3) + np.asarray(center)
    faces = grid_faces(rings, segments, close_cols=True)
    return verts, faces


def add_vertex_group(obj, name, weights):
    """Cree un groupe de vertices depuis un tableau de poids (un par sommet)."""
    weights = np.asarray(weights, dtype=np.float64)
    group = obj.vertex_groups.new(name=name)
    idx = np.nonzero(weights > 1e-4)[0]
    for i in idx:
        group.add([int(i)], float(min(1.0, weights[i])), "REPLACE")
    return group


def vertex_coords(obj):
    n = len(obj.data.vertices)
    arr = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", arr)
    return arr.reshape(n, 3)


def set_vertex_coords(obj, coords):
    obj.data.vertices.foreach_set("co", np.asarray(coords, dtype=np.float32).ravel())
    obj.data.update()


def shade_smooth(obj, angle=None):
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if angle is not None and hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = angle


def add_subsurf(obj, levels=1, render_levels=2):
    mod = obj.modifiers.new("Subdivision", "SUBSURF")
    mod.levels = levels
    mod.render_levels = render_levels
    return mod


def add_solidify(obj, thickness, offset=-1.0):
    mod = obj.modifiers.new("Solidify", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = offset
    return mod


def delete_faces_by_mask(obj, keep_face_fn):
    """Supprime les faces pour lesquelles keep_face_fn(centre) est faux."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    doomed = [f for f in bm.faces if not keep_face_fn(np.array(f.calc_center_median()))]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def snap_boundary(obj, project):
    """Recale les sommets de bord sur un contour ideal.

    Decouper des faces laisse forcement un bord en escalier : on projette
    chaque sommet de la frontiere sur la courbe voulue pour retrouver un
    contour net (indispensable pour la fente des paupieres et la bouche).
    `project(p)` renvoie la position corrigee, ou None si le sommet n'est pas
    concerne.
    """
    with bmesh_edit(obj) as bm:
        for v in bm.verts:
            if not v.is_boundary:
                continue
            new = project(np.array(v.co))
            if new is not None:
                v.co = tuple(float(c) for c in new)
    return obj


def relax_region(obj, weight, iterations=3, factor=0.55, pin_boundary=True):
    """Lissage laplacien pondere, sur une zone seulement.

    Sert a effacer les marches d'escalier que laisse une decoupe de faces :
    on relaxe la couronne de sommets autour du trou en gardant le bord fixe.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    weights = {}
    for v in bm.verts:
        if pin_boundary and v.is_boundary:
            continue
        w = weight(np.array(v.co))
        if w > 1e-3:
            weights[v] = min(1.0, w)

    for _ in range(iterations):
        moved = {}
        for v, w in weights.items():
            neigh = [e.other_vert(v) for e in v.link_edges]
            if not neigh:
                continue
            avg = np.mean([list(n.co) for n in neigh], axis=0)
            moved[v] = np.array(v.co) * (1.0 - factor * w) + avg * (factor * w)
        for v, co in moved.items():
            v.co = tuple(float(c) for c in co)

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def extrude_boundary_inward(obj, select, displace):
    """Extrude un bord libre et rentre la nouvelle boucle : donne l'epaisseur
    d'un bord de paupiere ou d'une levre."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    edges = [e for e in bm.edges if e.is_boundary and select(np.array(e.verts[0].co))]
    if edges:
        ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
        for g in ret["geom"]:
            if isinstance(g, bmesh.types.BMVert):
                g.co = tuple(float(c) for c in displace(np.array(g.co)))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def bmesh_edit(obj):
    """Context manager minimaliste pour manipuler un objet via bmesh."""
    class _Ctx:
        def __enter__(self):
            self.bm = bmesh.new()
            self.bm.from_mesh(obj.data)
            return self.bm

        def __exit__(self, *exc):
            if exc[0] is None:
                self.bm.to_mesh(obj.data)
                obj.data.update()
            self.bm.free()
            return False

    return _Ctx()


def join_objects(objects, name=None):
    """Fusionne une liste d'objets dans le premier."""
    objects = [o for o in objects if o is not None]
    target = objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()
    if name:
        target.name = name
        target.data.name = name
    bpy.ops.object.select_all(action="DESELECT")
    return target


def curve_object(name, points, radius_profile=None, bevel_depth=0.0,
                 resolution=12, collection=None):
    """Cree une courbe Bezier/poly lissee, utilisee pour la paille et les branches."""
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = resolution
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 6
    curve.use_fill_caps = True

    spline = curve.splines.new("NURBS")
    spline.points.add(len(points) - 1)
    for i, p in enumerate(points):
        spline.points[i].co = (p[0], p[1], p[2], 1.0)
        if radius_profile is not None:
            spline.points[i].radius = float(radius_profile[i])
    spline.use_endpoint_u = True
    spline.order_u = min(4, len(points))

    obj = bpy.data.objects.new(name, curve)
    return link(obj, collection)
