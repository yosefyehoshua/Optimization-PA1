import numpy as np
import matplotlib.pyplot as plt
import os

def plot_contours(f, x_lim, y_lim, levels=20, title=None, save_path=None):
    """
    Plot the contour lines of a 2D function.

    Args:
        f (callable): The function to be plotted.
        x_lim (tuple): The limits for the x-axis.
        y_lim (tuple): The limits for the y-axis.
        levels (int): The number of contour levels (default: 20).
        title (str): The title for the plot (default: None).
        save_path (str): The path to save the plot (default: None).
    """
    x = np.linspace(x_lim[0], x_lim[1], 100)
    y = np.linspace(y_lim[0], y_lim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), False)[0]

    plt.figure()
    plt.contour(X, Y, Z, levels)
    plt.colorbar()
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    if title:
        plt.title(title)
    if save_path:
        if title:
            save_path = save_path + os.sep + title.replace(" ", "_").replace(":", "")
        else:
            save_path = save_path + os.sep + "contours"
    plt.show()

       
def plot_paths(f, paths, names, x_lim, y_lim, title=None, save_path=None):
    """
    Plot the iteration paths for different methods.

    Args:
       paths (list): A list of iteration paths (lists of points).
       names (list): A list of names for the methods.
       x_lim (tuple): The limits for the x-axis.
       y_lim (tuple): The limits for the y-axis.
       title (str): The title for the plot (default: None).
       save_path (str): The path to save the plot (default: None).
    """
    plt.figure()
    for path, name in zip(paths, names):
       x_values = [point[0] for point in path]
       y_values = [point[1] for point in path]
       plt.plot(x_values, y_values, label=name)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    # Add contours with opacity 0.5
    x = np.linspace(x_lim[0], x_lim[1], 100)
    y = np.linspace(y_lim[0], y_lim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]), False)[0]
    plt.contourf(X, Y, Z, alpha=0.5)
    plt.colorbar()
    if title:
        plt.title(title)
    if save_path:
        if title:
            save_path = save_path + os.sep + title.replace(" ", "_").replace(":", "")
        else:
            save_path = save_path + os.sep + "paths"
        plt.savefig(save_path + '.png')
    plt.show()

def plot_objective_values(obj_values, names, title=None, save_path=None):
    """
    Plot the objective function values at each iteration for different methods.

    Args:
       obj_values (list): A list of lists of objective function values for each method.
       names (list): A list of names for the methods.
       title (str): The title for the plot (default: None).
       save_path (str): The path to save the plot (default: None).
    """
    plt.figure()
    for values, name in zip(obj_values, names):
       iterations = range(len(values))
       plt.plot(iterations, values, label=name)

    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.yscale('log')
    plt.legend()
    if title:
       plt.title(title)
    if save_path:
        if title:
            save_path = save_path + os.sep + title.replace(" ", "_").replace(":", "")
        else:
            save_path = save_path + os.sep + "objective_values"
        plt.savefig(save_path + '.png')
    plt.show()
