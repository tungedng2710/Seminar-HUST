import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# Define the gradient of the Rosenbrock function
def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# Gradient Descent implementation
def gradient_descent(grad_func, start, lr=0.001, max_iter=10000, tol=1e-6):
    x = start
    path = [x.copy()]
    for _ in range(max_iter):
        grad = grad_func(x)
        x_new = x - lr * grad
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(path)

# Run BFGS optimization
start_point = np.array([-1.2, 1.0])
bfgs_result = minimize(rosenbrock, start_point, method='BFGS', jac=rosenbrock_grad, options={'disp': False})
bfgs_path = bfgs_result['x']

# Run Gradient Descent
gd_path = gradient_descent(rosenbrock_grad, start_point)

# Plotting the optimization paths
xlist = np.linspace(-2, 2, 400)
ylist = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(xlist, ylist)
Z = (1 - X)**2 + 100*(Y - X**2)**2
import os

# Define path to save the figure
output_path = "logs/rosenbrock_optimization.png"

# Plotting and saving the figure
plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
plt.plot(gd_path[:, 0], gd_path[:, 1], 'r.-', label='Gradient Descent')
plt.plot(start_point[0], start_point[1], 'ko', label='Start Point')
plt.plot(1, 1, 'k*', markersize=10, label='Minimum')
plt.plot(bfgs_result['x'][0], bfgs_result['x'][1], 'go', label='BFGS Result')
plt.title('Optimization of Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig(output_path)
output_path