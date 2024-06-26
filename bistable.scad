difference(){
    cube([50, 50, 5], true);
    cube([40, 40, 10], true);
}

translate([0,4,0]) {
    intersection() {
            cube([18, 20, 10], true);
    union() {
        cube([8, 20, 5], true);
        translate([0, -7, 0]) cube([50,0.75,5], true);
        translate([0, +7, 0]) cube([50,0.75,5], true);
    }
}
};


difference() {
    union(){
        translate([0, -7, 0]) cube([50,0.75,5], true);
        translate([0, +7, 0]) cube([50,0.75,5], true);
    }

    cube([30,30,8], true);
}


translate([12,9,0]) rotate([0,0,-30]) cube([8, 2, 5], true);

translate([-12,9,0]) rotate([0,0,30]) cube([8, 2, 5], true);

translate([12,-5,0]) rotate([0,0,-30]) cube([8, 2, 5], true);

translate([-12,-5,0]) rotate([0,0,30]) cube([8, 2, 5], true);