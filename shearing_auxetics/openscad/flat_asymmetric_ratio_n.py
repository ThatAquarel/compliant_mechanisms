import numpy as np

from solid import *
from solid.utils import *

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

xy_basis = np.arange(X_GRID)[:, np.newaxis] * [w, h]


def build_fillet(r, h):
    return cube([r, r, h]) - cylinder(r, h)


def build_linkage():
    linkage = cube([COMPLIANCE_LENGTH, COMPLIANCE_THICKNESS, THICKNESS])
    linkage += translate([COMPLIANCE_FILLET_RADIUS, -COMPLIANCE_FILLET_RADIUS, 0])(
        mirror([1, 0, 0])(build_fillet(COMPLIANCE_FILLET_RADIUS, THICKNESS))
    )
    linkage += translate(
        [COMPLIANCE_LENGTH - COMPLIANCE_FILLET_RADIUS, -COMPLIANCE_FILLET_RADIUS, 0]
    )(build_fillet(COMPLIANCE_FILLET_RADIUS, THICKNESS))

    return linkage


# build objects
a = cube(0)
x_linkage, y_linkage = build_linkage(), rotate([0, 0, 90])(build_linkage())


# base
for j in range(Y_GRID):
    if j % 2 == 0:
        ratio = 1
        y_linkage_offset = [COMPLIANCE_THICKNESS, Y_SIZE]
        y_linkage_model = y_linkage

        x_linkage_offset = [0, COMPLIANCE_THICKNESS]
        x_linkage_model = rotate([0, 0, 180])(x_linkage)
    else:
        ratio = ASYMMETRIC_RATIO
        y_linkage_offset = [
            X_SIZE - COMPLIANCE_THICKNESS,
            Y_SIZE * ratio + COMPLIANCE_LENGTH,
        ]
        y_linkage_model = rotate([0, 0, 180])(y_linkage)

        x_linkage_offset = [-COMPLIANCE_LENGTH, Y_SIZE * ratio - COMPLIANCE_THICKNESS]
        x_linkage_model = x_linkage

    xy_offset = [0, (j // 2) * rh + (j - j // 2) * h]

    for xy in xy_basis:
        a += translate([*(xy + xy_offset), 0])(
            cube([X_SIZE, Y_SIZE * ratio, THICKNESS])
        )

        # vertical linkages
        if (j + 1) < Y_GRID:
            a += translate([*(xy + xy_offset + y_linkage_offset), 0])(y_linkage_model)
            a += translate([*(xy + xy_offset + x_linkage_offset), 0])(x_linkage_model)


# intersect
region = cube(
    [
        X_GRID * w - COMPLIANCE_LENGTH,
        (Y_GRID // 2) * (h + rh) - COMPLIANCE_LENGTH + (X_GRID - 2) * h,
        THICKNESS,
    ]
)
a *= translate([0, h, 0])(region)
a = translate([0, -h, 0])(a)

scad_render_to_file(
    a,
    f"./shearing_auxetics/flat_asymmetric_ratio_{ASYMMETRIC_RATIO}_{X_GRID}_{Y_GRID}.scad",
    file_header="$fn=32;",
)
