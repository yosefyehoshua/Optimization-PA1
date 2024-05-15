import numpy as np


def minimize(method, f, x0, obj_tol, param_tol, max_iter, c1=0.01, c2=0.5):
    """
    Line search minimization with Wolfe conditions and backtracking.

    Args:
        method (callable): The minimization method (e.g., gradient_descent, newton_method).
        f (callable): The objective function to be minimized.
        x0 (np.ndarray): The starting point.
        obj_tol (float): The tolerance for the objective function value change.
        param_tol (float): The tolerance for the parameter change.
        max_iter (int): The maximum number of iterations.
        c1 (float): The constant for the Wolfe conditions (default: 0.01).
        c2 (float): The backtracking constant (default: 0.5).

    Returns:
        np.ndarray: The final point.
        float: The final objective function value.
        bool: A flag indicating success or failure.
    """
    x = x0.copy()
    iterations = []
    obj_values = []
    
    for i in range(max_iter):

        # Check termination conditions
        if i > 1:
            if check_convergence(obj_values, iterations, obj_tol, param_tol):
                return x, obj_value, True, iterations, obj_values
        
        alpha = 1.0

        if method == "Gradient Descent":
            obj_value, gradient, _ = f(x, False) # TODO: check hessian clac to only if needed
            obj_values.append(obj_value)
            iterations.append(x.copy())
            p = -gradient
        elif method == "Newton's Method":  # Newton method
            try:
                obj_value, gradient, hessian = f(x, True) # TODO: check hessian clac to only if needed
                obj_values.append(obj_value)
                iterations.append(x.copy())
                p = np.linalg.solve(hessian, -gradient)
                
            except np.linalg.LinAlgError:
                print("Singular Hessian encountered. Terminating.")
                return x, obj_value, False, iterations, obj_values
        else:
            print("Invalid method. Terminating.")
            return x, obj_value, False, iterations, obj_values
        
        # Print iteration details
        print(f"Iteration {i}: x = {x}, f(x) = {obj_value}") 
        obj_value_new, gradient_new, hessian = f(x + alpha * p, True)

        # Line search with Wolfe conditions and backtracking
        while obj_value_new > obj_value + c1 * alpha * np.dot(gradient, p) or \
              np.dot(gradient_new, p) < c2 * np.dot(gradient, p): # Wolfe condition with backtracking for step length search
            alpha *= c2
            obj_value_new, gradient_new, hessian = f(x + alpha * p, True)

        x += alpha * p

    print("Maximum iterations reached.")
    return x, obj_value, False, iterations, obj_values


def check_convergence(obj_values, iterations, obj_tol, param_tol):
    """
    Check convergence based on the change in objective function value and parameters.

    Args:
        obj_values (list): List of objective function values.
        iterations (list): List of parameter values.
        obj_tol (float): The tolerance for the objective function value change.
        param_tol (float): The tolerance for the parameter change.

    Returns:
        bool: True if convergence is reached, False otherwise.
    """
    obj_diff = abs(obj_values[-1] - obj_values[-2])
    param_diff = np.linalg.norm(iterations[-1] - iterations[-2])
    if obj_diff < obj_tol and param_diff < param_tol:
        print("Convergence reached.")
        return True
    return False
