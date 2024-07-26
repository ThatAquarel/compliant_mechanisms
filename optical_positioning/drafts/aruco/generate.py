import cv2
import numpy as np

# Specify the dictionary to use (6x6 with 250 markers)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


# Function to generate and save an ArUco marker
def generate_aruco_marker(id, size=200, border_bits=1):
    # Create an empty image to store the marker
    marker_image = np.zeros((size, size), dtype=np.uint8)

    # Generate the marker
    marker_image = cv2.aruco.generateImageMarker(
        aruco_dict, id, size, marker_image, border_bits
    )

    # Save the marker image
    cv2.imwrite(
        f"optical_positioning/drafts/aruco/markers/aruco_marker_{id}.png", marker_image
    )


# Generate markers with specified IDs
for marker_id in range(250):  # Change range as needed
    generate_aruco_marker(marker_id)

print("ArUco markers generated and saved.")
