import bpy
import bmesh
import math

# clear old objects
bpy.ops.object.mode_set(mode="OBJECT")

CUTOUTS_COLLECTION = "Cutouts"


def del_collection(coll):
    for c in coll.children:
        del_collection(c)
    bpy.data.collections.remove(coll, do_unlink=True)


try:
    for i in bpy.data.objects:
        if not i.name in ["Light", "Camera"]:
            with bpy.context.temp_override(selected_objects=[i]):
                bpy.ops.object.delete()

    del_collection(bpy.data.collections[CUTOUTS_COLLECTION])
except:
    pass

# parametric constants
EPSILON = 1e-5
CURVE_RESOLUTION = 16

COMPLIANCE_LENGTH = 3
COMPLIANCE_THICKNESS = 0.6
COMPLIANCE_FILLET_RADIUS = 1

X_SIZE, Y_SIZE, THICKNESS = 6, 15, 4
X_GRID, Y_GRID = 8, 5

ASYMMETRIC_RATIO = 2

w, h, rh = (
    X_SIZE + COMPLIANCE_LENGTH,
    Y_SIZE + COMPLIANCE_LENGTH,
    Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH,
)


def is_similar(a, b):
    return (a - EPSILON) <= b <= (a + EPSILON)


def primitive_of_size(x, y, z, origin=[0, 0, 0]):
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        enter_editmode=False,
        location=(x / 2, y / 2, z / 2),
        scale=(x, y, z),
    )
    bpy.context.scene.cursor.location = origin
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    return bpy.context.object


def build_cutout(x, y, z, f):
    cutout = primitive_of_size(x, y, z)

    bm = bmesh.new()
    bm.from_mesh(cutout.data)

    def select_fillet(edge):
        dx, dy = 0, 0

        for i, vert in enumerate(edge.verts):
            if i % 2 == 0:
                dx += vert.co.x
                dy += vert.co.y
            else:
                dx -= vert.co.x
                dy -= vert.co.y

        return is_similar(0, dx) and is_similar(0, dy)

    bevel_geom = [edge for edge in bm.edges if select_fillet(edge)]

    ret = bmesh.ops.bevel(
        bm,
        geom=bevel_geom,
        offset=f,
        segments=CURVE_RESOLUTION,
        profile=0.5,
        affect="EDGES",
    )
    del ret

    bm.to_mesh(cutout.data)
    bm.free()

    return cutout


cutout_collection = bpy.data.collections.new(CUTOUTS_COLLECTION)

x_cutout_0 = build_cutout(
    (X_SIZE - COMPLIANCE_THICKNESS) * 2 + COMPLIANCE_LENGTH,
    COMPLIANCE_LENGTH,
    THICKNESS,
    COMPLIANCE_FILLET_RADIUS,
)
bpy.ops.transform.translate(
    value=(X_SIZE + COMPLIANCE_LENGTH + COMPLIANCE_THICKNESS, Y_SIZE, 0)
)
x_cutout_0.hide_set(True)
cutout_collection.objects.link(x_cutout_0)

x_cutout_1 = build_cutout(
    (X_SIZE - COMPLIANCE_THICKNESS) * 2 + COMPLIANCE_LENGTH,
    COMPLIANCE_LENGTH,
    THICKNESS,
    COMPLIANCE_FILLET_RADIUS,
)
bpy.ops.transform.translate(
    value=(
        -X_SIZE - COMPLIANCE_LENGTH + COMPLIANCE_THICKNESS,
        Y_SIZE * ASYMMETRIC_RATIO,
        0,
    )
)
x_cutout_1.hide_set(True)
cutout_collection.objects.link(x_cutout_1)

y_cutout_0 = build_cutout(
    COMPLIANCE_LENGTH,
    Y_SIZE * (ASYMMETRIC_RATIO + 1) + COMPLIANCE_LENGTH - COMPLIANCE_THICKNESS * 2,
    THICKNESS,
    COMPLIANCE_FILLET_RADIUS,
)
bpy.ops.transform.translate(value=(X_SIZE, COMPLIANCE_THICKNESS, 0))
y_cutout_0.hide_set(True)
cutout_collection.objects.link(y_cutout_0)

y_cutout_1 = build_cutout(
    COMPLIANCE_LENGTH,
    Y_SIZE * (ASYMMETRIC_RATIO + 1) + COMPLIANCE_LENGTH - COMPLIANCE_THICKNESS * 2,
    THICKNESS,
    COMPLIANCE_FILLET_RADIUS,
)
bpy.ops.transform.translate(
    value=(
        X_SIZE * 2 + COMPLIANCE_LENGTH,
        -COMPLIANCE_LENGTH - Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_THICKNESS,
        0,
    )
)
y_cutout_1.hide_set(True)
cutout_collection.objects.link(y_cutout_1)

corner_cutout = primitive_of_size(
    X_SIZE + COMPLIANCE_LENGTH, Y_SIZE * (ASYMMETRIC_RATIO - 1), THICKNESS
)
bpy.ops.transform.translate(
    value=(X_SIZE + COMPLIANCE_LENGTH, Y_SIZE + COMPLIANCE_LENGTH, 0)
)
corner_cutout.hide_set(True)
cutout_collection.objects.link(corner_cutout)

base_block = primitive_of_size(
    (X_SIZE + COMPLIANCE_LENGTH) * 2,
    Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH,
    THICKNESS,
    origin=(X_SIZE * 2 + COMPLIANCE_LENGTH, 0, 0),
)

bpy.ops.object.modifier_add(type="BOOLEAN")
bpy.context.object.modifiers["Boolean"].operand_type = "COLLECTION"
bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
bpy.context.object.modifiers["Boolean"].solver = "EXACT"
bpy.context.object.modifiers["Boolean"].collection = cutout_collection
bpy.ops.object.modifier_apply(modifier="Boolean")

array_y = "array_y"
bpy.ops.object.modifier_add(type="ARRAY")
mod = bpy.context.object.modifiers.get("Array")
if mod:
    mod.name = array_y
bpy.context.object.modifiers[array_y].use_relative_offset = False
bpy.context.object.modifiers[array_y].use_constant_offset = True
bpy.context.object.modifiers[array_y].constant_offset_displace[0] = -18
bpy.context.object.modifiers[array_y].constant_offset_displace[1] = 15
bpy.context.object.modifiers[array_y].count = Y_GRID
bpy.ops.object.modifier_apply(modifier=array_y)

array_x = "array_x"
bpy.ops.object.modifier_add(type="ARRAY")
mod = bpy.context.object.modifiers.get("Array")
if mod:
    mod.name = array_x
bpy.context.object.modifiers[array_x].use_relative_offset = False
bpy.context.object.modifiers[array_x].use_constant_offset = True
bpy.context.object.modifiers[array_x].constant_offset_displace[0] = 9
bpy.context.object.modifiers[array_x].constant_offset_displace[1] = 18
bpy.context.object.modifiers[array_x].count = X_GRID
bpy.ops.object.modifier_apply(modifier=array_x)

bpy.ops.transform.translate(value=(-(X_SIZE * 2 + COMPLIANCE_LENGTH), 0, 0))

compliance_angle = math.atan(w / h)
bpy.ops.transform.rotate(value=math.pi / 2 - compliance_angle, orient_axis="Z")
bpy.ops.transform.rotate(value=-math.pi / 2, orient_axis="X")

compliance_hypotenuse = w / math.sin(compliance_angle)
circumference = compliance_hypotenuse * X_GRID
radius = circumference / (2 * math.pi)

bpy.ops.mesh.primitive_circle_add(
    radius=radius,
    enter_editmode=False,
    align="WORLD",
    location=(0, -radius, 0),
    scale=(1, 1, 1),
    vertices=(X_GRID * CURVE_RESOLUTION),
)
circle = bpy.context.object
bpy.context.scene.cursor.location = [0, 0, 0]
bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

bpy.ops.object.mode_set(mode="EDIT")
circle_mesh = bpy.context.object.data

bm = bmesh.from_edit_mesh(circle_mesh)
rip_index = 0

rip_vertex = bm.verts[:][rip_index]
new_vertex = bm.verts.new(rip_vertex.co)
bm.verts.index_update()

for i, edge in enumerate(rip_vertex.link_edges[:]):
    adjacent_vertex = edge.other_vert(rip_vertex)
    bm.edges.remove(edge)

    if i % 2 == 0:
        bm.edges.new([new_vertex, adjacent_vertex])
    else:
        bm.edges.new([rip_vertex, adjacent_vertex])

bmesh.update_edit_mesh(circle_mesh)

bpy.ops.object.mode_set(mode="OBJECT")
# bpy.ops.object.convert(target="CURVE")

# bpy.context.view_layer.objects.active = base_block
# bpy.ops.object.modifier_add(type="CURVE")
# bpy.context.object.modifiers["Curve"].object = circle
