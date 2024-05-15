import unittest
import numpy as np
from src.unconstrained_min import minimize
from src.utils import plot_contours, plot_paths, plot_objective_values
from tests.examples import quadratic_1, quadratic_2, quadratic_3, rosenbrock, linear, smooth_triangle
from matplotlib import pyplot as plt

class TestUnconstrainedMinimization(unittest.TestCase):
    def test_quadratic_1(self):
        x0 = np.array([1, 1]).astype(float)
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100

        print("\nTesting quadratic_1 function:")

        method_names = ["Gradient Descent", "Newton's Method"]
        paths = []
        obj_values = []

        for method in method_names:
            print("\nUsing", method)
            x, obj_value, success, iterations, f_obj_values = minimize(method, quadratic_1, x0, obj_tol, param_tol, max_iter)
            print(f"Final point: {x}, Objective value: {obj_value}, Method: {method}, Success: {success}")
            paths.append(iterations)
            obj_values.append(f_obj_values)

        plot_contours(quadratic_1, (-2, 2), (-2, 2), title="Quadratic Function 1 (Contour Lines as Circles)")
        plot_paths(quadratic_1, paths, method_names, (-2, 2), (-2, 2), title="Quadratic Function 1: Iteration Paths")
        plot_objective_values(obj_values, method_names, title="Quadratic Function 1: Objective Values")

    def test_quadratic_2(self):
        x0 = np.array([1, 1]).astype(float)
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100

        print("\nTesting quadratic_2 function:")

        method_names = ["Gradient Descent", "Newton's Method"]
        paths = []
        obj_values = []

        for method in method_names:
            print("\nUsing", method)
            x, obj_value, success, iterations, f_obj_values = minimize(method, quadratic_2, x0, obj_tol, param_tol, max_iter)
            print(f"Final point: {x}, Objective value: {obj_value}, Method: {method}, Success: {success}")
            paths.append(iterations)
            obj_values.append(f_obj_values)

        plot_contours(quadratic_2, (-2, 2), (-2, 2), title="Quadratic Function 2 (Contour Lines as Axis-Aligned Ellipses)")
        plot_paths(quadratic_2, paths, method_names, (-2, 2), (-2, 2, ), title="Quadratic Function 2: Iteration Paths")
        plot_objective_values(obj_values, method_names, title="Quadratic Function 2: Objective Values")

    def test_quadratic_3(self):
        x0 = np.array([1, 1]).astype(float)
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100

        print("\nTesting quadratic_3 function:")

        method_names = ["Gradient Descent", "Newton's Method"]
        paths = []
        obj_values = []

        for method in method_names:
            print("\nUsing", method)
            x, obj_value, success, iterations, f_obj_values = minimize(method, quadratic_3, x0, obj_tol, param_tol, max_iter)
            print(f"Final point: {x}, Objective value: {obj_value}, Method: {method}, Success: {success}")
            paths.append(iterations)
            obj_values.append(f_obj_values)

        plot_contours(quadratic_3, (-2, 2), (-2, 2), title="Quadratic Function 3 (Contour Lines as Rotated Ellipses)")
        plot_paths(quadratic_3, paths, method_names, (-2, 2), (-2, 2), title="Quadratic Function 3: Iteration Paths")
        plot_objective_values(obj_values, method_names, title="Quadratic Function 3: Objective Values")

    def test_rosenbrock(self):
        x0 = np.array([-1, 2]).astype(float)
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 10000

        print("\nTesting Rosenbrock function:")

        method_names = ["Gradient Descent", "Newton's Method"]
        paths = []
        obj_values = []

        for method in method_names:
            print("\nUsing", method)
            x, obj_value, success, iterations, f_obj_values = minimize(method, rosenbrock, x0, obj_tol, param_tol, max_iter)
            print(f"Final point: {x}, Objective value: {obj_value}, Method: {method}, Success: {success}")
            paths.append(iterations)
            obj_values.append(f_obj_values)

        plot_contours(rosenbrock, (-2, 2), (-1, 3), title="Rosenbrock Function (Contour Lines as Banana-Shaped Ellipses)")
        plot_paths(rosenbrock, paths, method_names, (-2, 2), (-1, 3), title="Rosenbrock Function: Iteration Paths")
        plot_objective_values(obj_values, method_names, title="Rosenbrock Function: Objective Values")
        
    def test_linear(self):
        x0 = np.array([1, 1]).astype(float)
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100

        print("\nTesting linear function:")

        method_names = ["Gradient Descent", "Newton's Method"]
        paths = []
        obj_values = []

        for method in method_names:
            print("\nUsing", method)
            x, obj_value, success, iterations, f_obj_values = minimize(method, linear, x0, obj_tol, param_tol, max_iter)
            print(f"Final point: {x}, Objective value: {obj_value}, Method: {method}, Success: {success}")
            paths.append(iterations)
            obj_values.append(f_obj_values)

        plot_contours(linear, (-2, 2), (-2, 2), title="Linear Function (Contour Lines as Straight Lines)")
        plot_paths(linear, paths, method_names, (-2, 2), (-2, 2), title="Linear Function: Iteration Paths")
        plot_objective_values(obj_values, method_names, title="Linear Function: Objective Values")

    def test_smooth_triangle(self):
        x0 = np.array([1, 1]).astype(float)
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100

        print("\nTesting smooth_triangle function:")

        method_names = ["Gradient Descent", "Newton's Method"]
        paths = []
        obj_values = []

        for method in method_names:
            print("\nUsing", method)
            x, obj_value, success, iterations, f_obj_values = minimize(method, smooth_triangle, x0, obj_tol, param_tol, max_iter)
            print(f"Final point: {x}, Objective value: {obj_value}, Method: {method}, Success: {success}")
            paths.append(iterations)
            obj_values.append(f_obj_values)

        plot_contours(smooth_triangle, (-2, 2), (-2, 2), title="Smooth Triangle Function (Contour Lines as Smoothed Corner Triangles)")
        plot_paths(smooth_triangle, paths, method_names, (-2, 2), (-2, 2), title="Smooth Triangle Function: Iteration Paths")
        plot_objective_values(obj_values, method_names, title="Smooth Triangle Function: Objective Values")


if __name__ == '__main__':
    unittest.main()