import cv2
import numpy as np

PIXELS_PER_INCH = 300
PIXELS_PER_CM = 300 / 2.54
PAGE_SIZE_INCH = 11, 8.5

page_size = np.multiply(PAGE_SIZE_INCH, PIXELS_PER_INCH).astype(np.int32)
output = np.ones((*page_size, 3), dtype=np.uint8) * 255

MARGIN = 1  # cm
ARUCO_INDICES = np.arange(20) + 16
ARUCO_WIDTH = 4  # cm

margin = int(MARGIN * PIXELS_PER_CM)
aruco_pixels = int(ARUCO_WIDTH * PIXELS_PER_CM)
aruco_len = aruco_pixels + margin

aruco_nx, aruco_ny = np.divide(page_size, aruco_len).astype(np.int32)

for i, n in enumerate(ARUCO_INDICES):
    marker = cv2.imread(
        f"optical_positioning/drafts/aruco/markers/aruco_marker_{n}.png"
    )
    marker = cv2.resize(marker, (aruco_pixels,) * 2, interpolation=cv2.INTER_LINEAR)

    xi, yi = i % aruco_nx, i // aruco_nx
    if yi >= aruco_ny:
        print(f"skip {i}")
        continue

    xi, yi = xi * aruco_len, yi * aruco_len

    output[
        xi + margin : aruco_pixels + xi + margin,
        yi + margin : aruco_pixels + yi + margin,
    ] = marker

cv2.imwrite(
    f"optical_positioning/drafts/aruco/generate_ngrid_{ARUCO_WIDTH}cm.png", output
)
