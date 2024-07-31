import numpy as np


def find_pose(reference_points, current_points):
    assert reference_points.shape == current_points.shape
    assert reference_points.shape[0] == 4  # Ensure there are 4 points

    # Compute the centroids of the reference and current points
    centroid_ref = np.mean(reference_points, axis=0)
    centroid_cur = np.mean(current_points, axis=0)

    # Center the points by subtracting the centroids
    ref_centered = reference_points - centroid_ref
    cur_centered = current_points - centroid_cur

    # Compute the covariance matrix
    H = ref_centered.T @ cur_centered

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = centroid_cur - R @ centroid_ref

    # Construct the affine transformation matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R
    affine_matrix[:3, 3] = t

    return affine_matrix


def transform_points(points, transformation_matrix):
    # Convert points to homogeneous coordinates
    n = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((n, 1))))

    # Apply the transformation matrix
    transformed_points = transformation_matrix @ homogeneous_points.T

    # Convert back to Cartesian coordinates
    transformed_points_cartesian = transformed_points[:3, :].T

    return transformed_points_cartesian


# Example reference points (corners of the square in the reference pose)
reference_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

# Example current points (corners of the square in the current pose)
current_points = np.array([[1, 1, 1], [2, 1, 1], [2, 2, 1], [1, 2, 1]])

# Find the pose (transformation matrix)
pose_matrix = find_pose(reference_points, current_points)

# Transform the reference points using the pose matrix
transformed_points = transform_points(reference_points, pose_matrix)

print("Affine transformation matrix (3D):\n", pose_matrix)
print("Transformed points:\n", transformed_points)
print("Current points:\n", current_points)

# Check if the transformed points match the current points
match = np.allclose(transformed_points, current_points)
print("Do the transformed points match the current points?", match)
