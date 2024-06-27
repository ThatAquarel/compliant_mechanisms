from solid import *
from solid.utils import *

COMPLIANCE_LENGTH = 3
COMPLIANCE_THICKNESS = 0.6
COMPLIANCE_FILLET_RADIUS = 0.5

THICKNESS = 4


def fillet(r, h):
    return cube([r, r, h]) - cylinder(r, h)


linkage = cube([COMPLIANCE_LENGTH, COMPLIANCE_THICKNESS, THICKNESS])
linkage += translate([COMPLIANCE_FILLET_RADIUS, -COMPLIANCE_FILLET_RADIUS, 0])(
    mirror([1, 0, 0])(fillet(COMPLIANCE_FILLET_RADIUS, THICKNESS))
)
linkage += translate(
    [COMPLIANCE_LENGTH - COMPLIANCE_FILLET_RADIUS, -COMPLIANCE_FILLET_RADIUS, 0]
)(fillet(COMPLIANCE_FILLET_RADIUS, THICKNESS))

scad_render_to_file(linkage, "./shearing_auxetics/linkage.scad", file_header="$fn=32;")
