import cv2
import numpy as np

# K1 = np.array(
#     [
#         [479.70111284, 0.0, 639.65263158],
#         [0.0, 482.45603718, 398.87140028],
#         [0.0, 0.0, 1.0],
#     ]
# )
# D1 = np.array([[3.1147561], [-73.02753967], [118.53702217], [32.53536911]])

# K1 = np.array(
#     [
#         [690.65661124, 0.0, 639.32079512],
#         [0.0, 689.4605006, 399.36525519],
#         [0.0, 0.0, 1.0],
#     ]
# )
# D1 = np.array([[-0.26426162], [0.73247824], [0.05607988], [-17.91802217]])

# K1 = np.array([[1000.0, 0.0, 639.5], [0.0, 1000.0, 399.5], [0.0, 0.0, 1.0]])

# D1 = np.array([[0.0], [0.0], [0.0], [0.0]])

# Open the camera streams
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap1.set(cv2.CAP_PROP_FPS, 120)


def undistort_fisheye(image, K, D):
    DIM = image.shape[:2][::-1]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        # K, None, np.eye(3), K, DIM, cv2.CV_16SC2
        K,
        D,
        np.eye(3),
        K,
        DIM,
        cv2.CV_16SC2,
    )
    undistorted_img = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return undistorted_img


while True:
    # Capture frames from each camera
    ret1, frame1 = cap1.read()

    if not ret1:
        print("Failed to capture images")
        break

    # Undistort the frames
    undistorted_frame1 = undistort_fisheye(frame1, K1, D1)

    # Show the undistorted frames

    cv2.resize(undistorted_frame1, (960, 600), interpolation=cv2.INTER_LINEAR)

    cv2.imshow(
        "Camera 1",
        cv2.hconcat(
            [
                cv2.resize(
                    undistorted_frame1, (960, 600), interpolation=cv2.INTER_LINEAR
                ),
                cv2.resize(frame1, (960, 600), interpolation=cv2.INTER_LINEAR),
            ]
        ),
    )

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture objects and close windows
cap1.release()
cv2.destroyAllWindows()
