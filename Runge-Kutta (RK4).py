import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_surface(func, x_range=(-1, 2), t_range=(-1, 2), resolution=100):
    """
    Grafica la superficie 3D de una función de dos variables.

    Parámetros:
    - func: función f(x, t) que recibe dos arrays (X, T) y devuelve Z.
    - x_range: tupla (xmin, xmax) para el eje x.
    - t_range: tupla (tmin, tmax) para el eje t.
    - resolution: número de puntos en cada eje.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    t = np.linspace(t_range[0], t_range[1], resolution)
    X, T = np.meshgrid(x, t)
    Z = func(X, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(x, t)')
    plt.title('Superficie 3D de la función')
    plt.show()


def plot_contour(func, x_range=(-1, 2), t_range=(-1, 2), resolution=100, levels=20):
    """
    Grafica los contornos (curvas de nivel) de una función de dos variables.

    Parámetros:
    - func: función f(x, t) que recibe dos arrays (X, T) y devuelve Z.
    - x_range: tupla (xmin, xmax) para el eje x.
    - t_range: tupla (tmin, tmax) para el eje t.
    - resolution: número de puntos en cada eje.
    - levels: número de curvas de nivel.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    t = np.linspace(t_range[0], t_range[1], resolution)
    X, T = np.meshgrid(x, t)
    Z = func(X, T)

    plt.contour(X, T, Z, levels=levels)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Contornos de la función')
    plt.colorbar(label='Valor de f(x, t)')
    plt.show()


# Ejemplo de uso:
if __name__ == "__main__":
    # Definir tu propia función aquí:
    ejemplo = lambda X, T: np.exp(X) - X * np.exp(T)

    # Graficar superficie 3D
    plot_surface(ejemplo, x_range=(-1, 2), t_range=(-1, 2), resolution=200)

    # Graficar contornos 2D
    plot_contour(ejemplo, x_range=(-1, 2), t_range=(-1, 2), resolution=200, levels=30)
