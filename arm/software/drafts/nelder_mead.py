import numpy as np
from scipy.optimize import minimize


# Define the objective function
def objective_function(vector):
    x, y, z = vector
    return (x - 1) ** 2 + (y - 2) ** 2 + (z - 3) ** 2


# Initial guess
initial_guess = [0.0, 0.0, 0.0]

# Perform the optimization
result = minimize(objective_function, initial_guess, method="Nelder-Mead")

print("The minimum is at:", result.x)
print("Minimum value of the objective function:", result.fun)
