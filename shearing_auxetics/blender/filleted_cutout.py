import bpy
import bmesh


EPSILON = 1e-5

bpy.ops.mesh.primitive_cube_add(
    size=1,
    enter_editmode=False,
    location=(0,) * 3,
    scale=(1,) * 3,
)

obj = bpy.context.object.data
bm = bmesh.new()
bm.from_mesh(obj)


def select_fillet(edge):
    dx, dy = 0, 0

    for i, vert in enumerate(edge.verts):
        if i % 2 == 0:
            dx += vert.co.x
            dy += vert.co.y
        else:
            dx -= vert.co.x
            dy -= vert.co.y

    return -EPSILON <= dx <= EPSILON and -EPSILON <= dy <= EPSILON


bevel_geom = [edge for edge in bm.edges if select_fillet(edge)]

ret = bmesh.ops.bevel(
    bm,
    geom=bevel_geom,
    offset=0.25,
    segments=8,
    profile=0.5,
    affect="EDGES",
)

bm.to_mesh(obj)
bm.free()
