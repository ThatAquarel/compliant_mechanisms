import numpy as np

# normalized to 1 vector endpoints
a = [[3, 2, 1], [7, -2, 1], [6, 0, 1]]
b = [[-2, 7], [2, 3], [0, 6]]

# a = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
# b = [[-2, 0], [-1, -1], [-2, -1]]

T, _, _, _ = np.linalg.lstsq(a, b)
# T = np.linalg.solve(a, b)
T = np.concatenate((T, [[0], [0], [1]]), axis=-1)

T = T.T

tx = T[0, 2]
ty = T[1, 2]

# Extract scaling and shear
a, b = T[0, 0], T[0, 1]
c, d = T[1, 0], T[1, 1]

# Compute scaling factors
sx = np.sqrt(a**2 + c**2)
sy = np.sqrt(b**2 + d**2)

# Compute shear
shear = (a * b + c * d) / (sx * sy)

# Compute rotation angle
theta = np.arctan2(c, a)

print(f"Translation: tx={tx}, ty={ty}")
print(f"Scaling: sx={sx}, sy={sy}")
print(f"Shear: {shear}")
print(f"Rotation (radians): {theta}")
print(f"Rotation (degrees): {np.degrees(theta)}")

# print(a)
# print(b)
# print(transform_ab)
# print(a @ transform_ab)
# print(b @ np.linalg.inv(transform_ab))
# print(np.dot([0, 0, 1], transform_ab))
# print(np.dot([5, 0, 1], transform_ab))
# print(np.dot([4, 1, 1], transform_ab))

# c = [[-2, 2], [2, -2]]
