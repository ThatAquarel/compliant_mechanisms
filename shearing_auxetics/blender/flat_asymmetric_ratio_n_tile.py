import bpy
import bmesh
import math

# clear old objects

CUTOUTS_COLLECTION, GRID_CUTOUTS_COLLECTION, FLANGES_COLLECTION = (
    "Cutouts",
    "GridCutouts",
    "Flanges",
)


def del_collection(coll):
    for c in coll.children:
        del_collection(c)
    bpy.data.collections.remove(coll, do_unlink=True)


names_filter = ["Light", "Camera", "Point", "Plane"]

try:
    for i in bpy.data.objects:
        if any(i.name in name for name in names_filter):
            continue
        with bpy.context.temp_override(selected_objects=[i]):
            bpy.ops.object.delete()
except:
    pass
try:
    del_collection(bpy.data.collections[CUTOUTS_COLLECTION])
except:
    pass
try:
    del_collection(bpy.data.collections[FLANGES_COLLECTION])
except:
    pass
try:
    del_collection(bpy.data.collections[GRID_CUTOUTS_COLLECTION])
except:
    pass

# parametric constants
EPSILON = 1e-5
CURVE_RESOLUTION = 16
REMESH = True
REMESH_RESOLUTION = 9

COMPLIANCE_LENGTH = 3
COMPLIANCE_THICKNESS = 1.1
COMPLIANCE_FILLET_RADIUS = COMPLIANCE_LENGTH / 2

X_SIZE, Y_SIZE, THICKNESS = 6, 15, 2
X_GRID, Y_GRID = 4, 4

ASYMMETRIC_RATIO = 2.5

SUPPORT_WIDTH, SUPPORT_HEIGHT = COMPLIANCE_THICKNESS * 2, COMPLIANCE_LENGTH
FLANGE_HEIGHT, FLANGE_THICKNESS = 7, THICKNESS

# TILE_BOTTOM, TILE_CENTER, TILE_TOP = True, False, False
# TILE_BOTTOM, TILE_CENTER, TILE_TOP = False, True, False
TILE_BOTTOM, TILE_CENTER, TILE_TOP = False, False, True

w, h, rh = (
    X_SIZE + COMPLIANCE_LENGTH,
    Y_SIZE + COMPLIANCE_LENGTH,
    Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH,
)


def hide(obj):
    obj.hide_set(True)
    obj.hide_render = True


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


def array_offset(
    x,
    y,
    z,
    n,
    name="Array",
):
    bpy.ops.object.modifier_add(type="ARRAY")
    mod = bpy.context.object.modifiers.get("Array")
    if mod:
        mod.name = name
    bpy.context.object.modifiers[name].use_relative_offset = False
    bpy.context.object.modifiers[name].use_constant_offset = True
    bpy.context.object.modifiers[name].constant_offset_displace = (x, y, z)
    bpy.context.object.modifiers[name].count = n
    bpy.ops.object.modifier_apply(modifier=name)


def boolean(x, operation="UNION", operand_type="OBJECT", name="Boolean"):
    bpy.ops.object.modifier_add(type="BOOLEAN")
    mod = bpy.context.object.modifiers.get("Boolean")
    if mod:
        mod.name = name
    bpy.context.object.modifiers[name].operand_type = operand_type
    bpy.context.object.modifiers[name].operation = operation
    bpy.context.object.modifiers[name].solver = "EXACT"
    if operand_type == "OBJECT":
        bpy.context.object.modifiers[name].object = x
    else:
        bpy.context.object.modifiers[name].collection = x
    bpy.ops.object.modifier_apply(modifier=name)


def collection_new(name, boolean_proof=False):
    collection = bpy.data.collections.new(name)
    if boolean_proof:
        collection.objects.link(primitive_of_size(0, 0, 0))
    return collection


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
hide(x_cutout_0)
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
hide(x_cutout_1)
cutout_collection.objects.link(x_cutout_1)

y_cutout_0 = build_cutout(
    COMPLIANCE_LENGTH,
    Y_SIZE * (ASYMMETRIC_RATIO + 1) + COMPLIANCE_LENGTH - COMPLIANCE_THICKNESS * 2,
    THICKNESS,
    COMPLIANCE_FILLET_RADIUS,
)
bpy.ops.transform.translate(value=(X_SIZE, COMPLIANCE_THICKNESS, 0))
hide(y_cutout_0)
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
hide(y_cutout_1)
cutout_collection.objects.link(y_cutout_1)

corner_cutout = primitive_of_size(
    X_SIZE + COMPLIANCE_LENGTH, Y_SIZE * (ASYMMETRIC_RATIO - 1), THICKNESS
)
bpy.ops.transform.translate(
    value=(X_SIZE + COMPLIANCE_LENGTH, Y_SIZE + COMPLIANCE_LENGTH, 0)
)
hide(corner_cutout)
cutout_collection.objects.link(corner_cutout)

compliance_angle = math.atan(w / h)
compliance_hypotenuse = w / math.sin(compliance_angle)

if not TILE_CENTER:
    flanges_collection = collection_new(FLANGES_COLLECTION, boolean_proof=True)
    grid_cutout_collection = collection_new(GRID_CUTOUTS_COLLECTION, boolean_proof=True)

if TILE_BOTTOM:
    support_0 = primitive_of_size(SUPPORT_WIDTH, SUPPORT_HEIGHT, THICKNESS)
    bpy.ops.transform.translate(
        value=(-SUPPORT_WIDTH / 2, -SUPPORT_HEIGHT + SUPPORT_WIDTH, 0)
    )
    flange_0 = primitive_of_size(compliance_hypotenuse, FLANGE_HEIGHT, FLANGE_THICKNESS)
    bpy.ops.transform.translate(
        value=(-SUPPORT_WIDTH / 2, -FLANGE_HEIGHT - SUPPORT_HEIGHT + SUPPORT_WIDTH, 0)
    )
    boolean(support_0)
    array_offset(compliance_hypotenuse, 0, 0, X_GRID, name="flange_0")
    hide(support_0)
    hide(flange_0)
    flanges_collection.objects.link(flange_0)
if TILE_TOP:
    support_1 = primitive_of_size(SUPPORT_WIDTH, SUPPORT_HEIGHT, THICKNESS)
    x_skew = -X_SIZE * math.cos(math.pi / 2 - compliance_angle) + (
        (Y_SIZE * (ASYMMETRIC_RATIO + 1) + COMPLIANCE_LENGTH * 2) * (Y_GRID - 1)
        + Y_SIZE * ASYMMETRIC_RATIO
    ) * math.cos(compliance_angle)
    # x_skew = x_skew % compliance_hypotenuse # put flange closest to axis
    x_skew -= (Y_GRID - 1) * 2 * compliance_hypotenuse
    y_skew = X_SIZE * math.sin(math.pi / 2 - compliance_angle) + (
        (Y_SIZE * (ASYMMETRIC_RATIO + 1) + COMPLIANCE_LENGTH * 2) * (Y_GRID - 1)
        + Y_SIZE * ASYMMETRIC_RATIO
    ) * math.sin(compliance_angle)
    bpy.ops.transform.translate(
        value=(x_skew - SUPPORT_WIDTH / 2, y_skew - SUPPORT_WIDTH, 0)
    )
    flange_1 = primitive_of_size(compliance_hypotenuse, FLANGE_HEIGHT, FLANGE_THICKNESS)
    bpy.ops.transform.translate(
        value=(x_skew - SUPPORT_WIDTH / 2, y_skew + SUPPORT_HEIGHT - SUPPORT_WIDTH, 0)
    )
    boolean(support_1)
    array_offset(compliance_hypotenuse, 0, 0, X_GRID, name="flange_1")
    hide(support_1)
    hide(flange_1)
    flanges_collection.objects.link(flange_1)

if TILE_BOTTOM:
    bottom_grid_cutout = primitive_of_size(
        X_SIZE + COMPLIANCE_LENGTH * 2, Y_SIZE + COMPLIANCE_LENGTH, THICKNESS
    )
    bpy.ops.transform.translate(value=(X_SIZE, 0, 0))
    hide(bottom_grid_cutout)
    grid_cutout_collection.objects.link(bottom_grid_cutout)

if TILE_TOP:
    top_grid_cutout = primitive_of_size(X_SIZE, COMPLIANCE_LENGTH, THICKNESS)
    bpy.ops.transform.translate(
        value=(
            -(Y_GRID - 1) * (X_SIZE * 2 + COMPLIANCE_LENGTH * 2),
            (Y_GRID - 1) * (rh - h) + Y_SIZE * ASYMMETRIC_RATIO,
            0,
        )
    )
    hide(top_grid_cutout)
    grid_cutout_collection.objects.link(top_grid_cutout)

base_block = primitive_of_size(
    (X_SIZE + COMPLIANCE_LENGTH) * 2,
    Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH,
    THICKNESS,
    origin=(X_SIZE, 0, 0),
)

boolean(cutout_collection, operation="DIFFERENCE", operand_type="COLLECTION")

array_offset(
    -(X_SIZE + COMPLIANCE_LENGTH) * 2,
    Y_SIZE * (ASYMMETRIC_RATIO - 1),
    0,
    Y_GRID,
    name="array_y",
)

if not TILE_CENTER:
    boolean(grid_cutout_collection, operation="DIFFERENCE", operand_type="COLLECTION")

array_offset(
    X_SIZE + COMPLIANCE_LENGTH, Y_SIZE + COMPLIANCE_LENGTH, 0, X_GRID, name="array_x"
)

bpy.ops.transform.translate(value=(-X_SIZE, 0, 0))
bpy.ops.transform.rotate(value=math.pi / 2 - compliance_angle, orient_axis="Z")
if not TILE_CENTER:
    boolean(flanges_collection, operation="UNION", operand_type="COLLECTION")
bpy.ops.transform.rotate(value=-math.pi / 2, orient_axis="X")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
min_x = min([i[0] for i in bpy.context.object.bound_box])
bpy.ops.transform.translate(value=(-min_x, 0, 0))

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
bpy.context.scene.cursor.location = [0, -THICKNESS, 0]
bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

bpy.ops.object.mode_set(mode="EDIT")
circle_mesh = bpy.context.object.data

bm = bmesh.from_edit_mesh(circle_mesh)

connection_vert = bm.verts[:][-1]
new_verts = [bm.verts.new(vert.co) for vert in bm.verts[:]]
new_verts.append(bm.verts.new(new_verts[0].co))
bm.verts.index_update()

bm.edges.remove(bm.edges[:][-1])
bm.edges.new([connection_vert, new_verts[0]])
new_edges = [
    bm.edges.new([vert, new_verts[i + 1]]) for i, vert in enumerate(new_verts[:-1])
]
bm.edges.index_update()

bmesh.update_edit_mesh(circle.data)

bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.convert(target="CURVE")

bpy.context.view_layer.objects.active = base_block
if REMESH:
    bpy.ops.object.modifier_add(type="REMESH")
    bpy.context.object.modifiers["Remesh"].mode = "SHARP"
    bpy.context.object.modifiers["Remesh"].octree_depth = REMESH_RESOLUTION
    bpy.ops.object.modifier_apply(modifier="Remesh")

bpy.ops.object.modifier_add(type="CURVE")
bpy.context.object.modifiers["Curve"].object = circle
