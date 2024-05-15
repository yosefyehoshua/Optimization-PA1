import numpy as np

def quadratic_1(x, calc_hessian=False):
    """
    f(x) = x^T Q x, where Q = [[1, 0], [0, 1]]
    """
    Q = np.array([[1, 0], [0, 1]])
    f = x.T.dot(Q).dot(x)
    g = (Q + Q.T).dot(x)
    if calc_hessian:
        h = Q + Q.T
    else:
        h = None
    return f, g, h


def quadratic_2(x, calc_hessian=False):
    """
    f(x) = x^T Q x, where Q = [[1, 0], [0, 100]]
    """
    Q = np.array([[1, 0], [0, 100]])
    f = x.T.dot(Q).dot(x)
    g = (Q + Q.T).dot(x)
    if calc_hessian:
        h = Q + Q.T
    else:
        h = None
    return f, g, h

def quadratic_3(x, calc_hessian=False):
    """
    f(x) = x^T Q x, where Q = R^T [[100, 0], [0, 1]] R
    and R is a rotation matrix.
    """
    R = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q_1 = np.array([[100, 0], [0, 1]])
    Q = R.T.dot(Q_1).dot(R)
    f = x.T.dot(Q).dot(x)
    g = (Q + Q.T).dot(x)
    if calc_hessian:
        h = Q + Q.T
    else:
        h = None
    return f, g, h

def rosenbrock(x, calc_hessian=False):
    """
    Rosenbrock function with contour lines as banana-shaped ellipses.
    f(x) = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    """
    x_1, x_2 = x[0], x[1]
    f = 100 * (x_2 - x_1**2)**2 + (1 - x_1)**2
    g = g = np.array([400 * x_1**3 - 400 * x_1 * x_2 + 2 * x_1 -2, -200 * x_1**2 + 200*x_2])
    if calc_hessian:
        h = np.array([[1200 * x_1**2 - 400 * x_2 + 2, -400 * x_1],[-400 * x_1, 200]])
    else:
        h = None
    return f, g, h


def linear(x, calc_hessian=False):
    """
    f(x) = a^T x, where a is [1, 2].
    """
    a = np.array([1, 2])
    f = a.dot(x)
    g = a
    h = 0
    return f, g, h

def smooth_triangle(x, calc_hessian=False):
    """
    f(x) = exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) + exp(-x[0] - 0.1)
    """
    x_1, x_2 = x[0], x[1]
    f = np.exp(x_1 + 3*x_2 - 0.1) + np.exp(x_1 - 3*x_2 - 0.1) + np.exp(-x_1 - 0.1)
    g = np.array([np.exp(x_1 + 3*x_2 - 0.1) + np.exp(x_1 - 3*x_2 - 0.1) - np.exp(-x_1 - 0.1), 
                  3*np.exp(x_1 + 3*x_2 - 0.1) - 3*np.exp(x_1 - 3*x_2 - 0.1)])
    if calc_hessian:
        h = np.array([[np.exp(x_1 + 3*x_2 - 0.1) + np.exp(x_1 - 3*x_2 - 0.1) + np.exp(-x_1 - 0.1) ,
                       3 * np.exp(x_1 + 3*x_2 - 0.1) - 3 * np.exp(x_1 - 3*x_2 - 0.1)],
                      [ 3 * np.exp(x_1 + 3*x_2 - 0.1) - 3 * np.exp(x_1 - 3*x_2 - 0.1), 
                       9 * np.exp(x_1 + 3*x_2 - 0.1) + 9 * np.exp(x_1 - 3*x_2 - 0.1)]])
    else:
        h = None
        
    return f, g, h