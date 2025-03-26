import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Define the function, its gradient, and its Hessian
# ------------------------------------------------

def f(xy):
    """
    Quadratic function: f(x, y) = x^2 + 4y^2
    """
    x, y = xy
    return x**2 + 4*y**2

def grad_f(xy):
    """
    Gradient of f(x, y).
    ∇f(x, y) = [2x, 8y]
    """
    x, y = xy
    return np.array([2*x, 8*y], dtype=float)

def hess_f(xy):
    """
    Hessian of f(x, y).
    Hessian = [[2,   0 ],
               [0,   8 ]]
    """
    # For this quadratic function, the Hessian is constant.
    return np.array([[2.0, 0.0],
                     [0.0, 8.0]], dtype=float)

# ------------------------------------------------
# Gradient Descent
# ------------------------------------------------
def gradient_descent(x0, step_size=0.1, tol=1e-6, max_iter=100):
    """
    Performs Gradient Descent starting at x0.
    Returns:
      xs  -> list of parameter (x, y) at each iteration
      fs  -> list of f(x, y) values at each iteration
    """
    xk = np.array(x0, dtype=float)
    xs, fs = [], []
    
    for _ in range(max_iter):
        xs.append(xk.copy())
        fs.append(f(xk))
        
        # Compute gradient
        gk = grad_f(xk)
        
        # Update rule: x_{k+1} = x_k - alpha * grad_f(x_k)
        xk_new = xk - step_size * gk
        
        # Check for convergence
        if np.linalg.norm(xk_new - xk) < tol:
            xk = xk_new
            break
        xk = xk_new
    
    # Capture final iteration
    xs.append(xk.copy())
    fs.append(f(xk))
    
    return xs, fs

# ------------------------------------------------
# Newton's Method
# ------------------------------------------------
def newton_method(x0, tol=1e-6, max_iter=100):
    """
    Performs Newton's Method starting at x0.
    Returns:
      xs  -> list of parameter (x, y) at each iteration
      fs  -> list of f(x, y) values at each iteration
    """
    xk = np.array(x0, dtype=float)
    xs, fs = [], []
    
    for _ in range(max_iter):
        xs.append(xk.copy())
        fs.append(f(xk))
        
        # Compute gradient and Hessian
        gk = grad_f(xk)
        Hk = hess_f(xk)
        
        # Solve Hk * p = gk for p
        # Newton direction p = H^{-1} * grad_f(xk)
        # Newton update: x_{k+1} = x_k - H^{-1}(x_k)*grad_f(x_k)
        p = np.linalg.solve(Hk, gk)
        xk_new = xk - p
        
        # Check for convergence
        if np.linalg.norm(xk_new - xk) < tol:
            xk = xk_new
            break
        xk = xk_new
    
    # Capture final iteration
    xs.append(xk.copy())
    fs.append(f(xk))
    
    return xs, fs


# ------------------------------------------------
# Run both methods
# ------------------------------------------------

# Initial guess
x0 = [2.0, -1.0]

# Run Gradient Descent
gd_xs, gd_fs = gradient_descent(x0, step_size=0.1, tol=1e-8, max_iter=50)

# Run Newton's Method
nm_xs, nm_fs = newton_method(x0, tol=1e-8, max_iter=50)

# ------------------------------------------------
# Plot the results
# ------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(gd_fs, label='Gradient Descent')
plt.plot(nm_fs, label='Newton Method', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('f(x, y)')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('logs/comparison_newton_gradient.png', dpi=150)

plt.show()