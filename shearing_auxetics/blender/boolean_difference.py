import bpy

bpy.ops.mesh.primitive_cube_add(
    size=2, enter_editmode=False, align="WORLD", location=(1, 0, 0), scale=(1, 1, 1)
)
o = bpy.context.object

bpy.ops.mesh.primitive_cube_add(
    size=2, enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
)
bpy.ops.object.modifier_add(type="BOOLEAN")
bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
bpy.context.object.modifiers["Boolean"].solver = "EXACT"
bpy.context.object.modifiers["Boolean"].object = o
