import numpy as np
import scipy.optimize

# Define the data points
data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0]])


# Define the cost function
def curve_fitting_cost(params, x, y):
    a, b, c = params
    return y - (a * x**2 + b * x + c)


# Extract x and y data
x_data = data[:, 0]
y_data = data[:, 1]

# Initial guess for the parameters (a, b, c)
initial_params = np.array([0.0, 0.0, 0.0])

# Use least_squares to minimize the cost function
result = scipy.optimize.least_squares(
    curve_fitting_cost, initial_params, args=(x_data, y_data)
)

# Extract the optimized parameters
optimized_params = result.x

# Output the results
print("Initial parameters: a = 0.0, b = 0.0, c = 0.0")
print(
    f"Optimized parameters: a = {optimized_params[0]}, b = {optimized_params[1]}, c = {optimized_params[2]}"
)
