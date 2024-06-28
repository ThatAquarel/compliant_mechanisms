import numpy as np


# parametric constants
# units: mm

COMPLIANCE_LENGTH = 3
COMPLIANCE_THICKNESS = 0.6
COMPLIANCE_FILLET_RADIUS = 1

# X_SIZE, Y_SIZE, THICKNESS = 6, 6, 4
# X_GRID, Y_GRID = 12, 7
# ASYMMETRIC_RATIO = 5

X_SIZE, Y_SIZE, THICKNESS = 6, 15, 4
X_GRID, Y_GRID = 8, 5
ASYMMETRIC_RATIO = 2

# programmatic constants

w, h, rh = (
    X_SIZE + COMPLIANCE_LENGTH,
    Y_SIZE + COMPLIANCE_LENGTH,
    Y_SIZE * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH,
)

compliance_angle = np.atan(w / h)
compliance_hypotenuse = w / np.sin(compliance_angle)

# circumscribed circle to polygon

slice_angle = (2 * np.pi) / X_GRID
circumference = compliance_hypotenuse * X_GRID
radius = circumference / (2 * np.pi)

assert Y_GRID % 2 == 1
height = (Y_GRID // 2) * rh + (Y_GRID - Y_GRID // 2) * h + (Y_SIZE / w) * X_SIZE
height *= np.sin(compliance_angle)


# compute gap offset
x_x_offset = (X_SIZE + COMPLIANCE_LENGTH / 2) * np.cos(np.pi / 2 - compliance_angle)
x_x_offset += (COMPLIANCE_LENGTH / 2) * np.cos(compliance_angle)
x_rad_offset = x_x_offset / radius
x_y_offset = (X_SIZE + COMPLIANCE_LENGTH / 2) * np.sin(np.pi / 2 - compliance_angle)
x_y_offset -= (COMPLIANCE_LENGTH / 2) * np.sin(compliance_angle)
x_ver_offset = x_y_offset

# build

from solid import *
from solid.utils import *

base_radians = np.linspace(0, 2 * np.pi, X_GRID, endpoint=False)

a = cylinder(radius, height)
a -= cylinder(radius - THICKNESS, height)


def build_gap(x, y, z, f):
    b, h = x - f * 2, y - f * 2
    assert b > 0 and h > 0
    c = translate([-b / 2, -h / 2])(square([b, h]))
    c = offset(f)(c)
    c = linear_extrude(z)(c)
    return c


x_gap = build_gap(
    X_SIZE * 2 + COMPLIANCE_LENGTH - COMPLIANCE_THICKNESS * 2,
    COMPLIANCE_LENGTH,
    radius,
    COMPLIANCE_FILLET_RADIUS,
)
y_gap = build_gap(
    (Y_SIZE + 1) * ASYMMETRIC_RATIO + COMPLIANCE_LENGTH - COMPLIANCE_THICKNESS * 2,
    COMPLIANCE_LENGTH,
    radius,
    COMPLIANCE_FILLET_RADIUS,
)

for j in range(Y_GRID):
    if j % 2 == 0:
        gap = x_gap
        rx = compliance_angle
        rdz = -x_rad_offset
        tdz = x_ver_offset
    else:
        gap = y_gap
        rx = -(np.pi / 2 - compliance_angle)
        rdz = 0
        tdz = 0

    gap = rotate([0, 90, 0])(gap)

    for t in base_radians:
        b = rotate([rx * 180 / np.pi, 0, 0])(gap)
        b = rotate([0, 0, (t + rdz) * 180 / np.pi])(b)
        b = up(j * Y_SIZE + tdz)(b)

        if j % 2 == 0:
            a -= b


scad_render_to_file(
    a,
    f"./shearing_auxetics/cylinder_asymmetric_ratio_{ASYMMETRIC_RATIO}_{X_GRID}_{Y_GRID}.scad",
    file_header="$fn=32;",
)
