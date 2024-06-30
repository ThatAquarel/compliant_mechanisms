import bpy
import numpy as np

for i in bpy.context.scene.objects:
    if not i.name in ["Light", "Camera"]:
        with bpy.context.temp_override(selected_objects=[i]):
            bpy.ops.object.delete()

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


base_size = np.array(
    [
        (X_SIZE + COMPLIANCE_LENGTH) * 2,
        Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH,
        THICKNESS,
    ]
)
bpy.ops.mesh.primitive_cube_add(
    size=1,
    enter_editmode=False,
    location=base_size / 2,
    scale=base_size,
)
bpy.context.scene.cursor.location = [0, 0, 0]
bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

base_block = bpy.context.object
