from solid import *
from solid.utils import *


COMPLIANCE_LENGTH = 3
COMPLIANCE_THICKNESS = 0.6
COMPLIANCE_FILLET_RADIUS = 0.5

X_SIZE, Y_SIZE, THICKNESS = 15, 15, 4
X_GRID, Y_GRID = 4, 4


a = cube(0)


def fillet(r, h):
    return cube([r, r, h]) - cylinder(r, h)


def coords(x, y):
    return x * (X_SIZE + COMPLIANCE_LENGTH), y * (Y_SIZE + COMPLIANCE_LENGTH)


# base
for i in range(X_GRID):
    for j in range(Y_GRID):
        a += translate([*coords(i, j), 0])(cube([X_SIZE, Y_SIZE, THICKNESS]))

# horizontal linkages
for i in range(1, X_GRID, 2):
    for j in range(Y_GRID):
        x, y = coords(i, j)
        linkage = cube([COMPLIANCE_LENGTH, COMPLIANCE_THICKNESS, THICKNESS])
        linkage += mirror([1, 0, 0])(
            translate([-COMPLIANCE_FILLET_RADIUS, -COMPLIANCE_FILLET_RADIUS, 0])(
                fillet(COMPLIANCE_FILLET_RADIUS, THICKNESS)
            )
        )
        linkage += translate(
            [COMPLIANCE_LENGTH - COMPLIANCE_FILLET_RADIUS, -COMPLIANCE_FILLET_RADIUS, 0]
        )(fillet(COMPLIANCE_FILLET_RADIUS, THICKNESS))

        linkage_mirror = translate([0, COMPLIANCE_THICKNESS])(
            mirror([0, 1, 0])(linkage)
        )

        if j % 2 == 0:
            a += translate(
                [x - COMPLIANCE_LENGTH, y + Y_SIZE - COMPLIANCE_THICKNESS, 0]
            )(linkage)
            b = translate([x + X_SIZE, y, 0])(linkage_mirror)
        else:
            a += translate([x - COMPLIANCE_LENGTH, y, 0])(linkage_mirror)
            b = translate([x + X_SIZE, y + Y_SIZE - COMPLIANCE_THICKNESS, 0])(linkage)
        if i + 1 < X_GRID:
            a += b

# vertical linkages
for i in range(X_GRID):
    for j in range(1, Y_GRID, 2):
        x, y = coords(i, j)
        linkage = cube([COMPLIANCE_THICKNESS, COMPLIANCE_LENGTH, THICKNESS])

        if i % 2 == 0:
            a += translate([x, y - COMPLIANCE_LENGTH, 0])(linkage)
            b = translate([x + X_SIZE - COMPLIANCE_THICKNESS, y + Y_SIZE, 0])(linkage)
        else:
            a += translate(
                [x + X_SIZE - COMPLIANCE_THICKNESS, y - COMPLIANCE_LENGTH, 0]
            )(linkage)
            b = translate([x, y + Y_SIZE, 0])(linkage)
        if j + 1 < Y_GRID:
            a += b


scad_render_to_file(
    a, "./shearing_auxetics/flat_ratio_equal.scad", file_header="$fn=32;"
)
