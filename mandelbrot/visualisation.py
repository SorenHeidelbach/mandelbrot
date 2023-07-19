import matplotlib.pyplot as plt
from mandelbrot.mandelbrot import mandelbrot_set_numba_parallel


def plot_mandelbrot_set(mandelbort_set , vmin = 0, vmax=1, extent=[-2, 1, -1.5, 1.5]):
    """
    Plots the Mandelbrot set for precalculated values
    """
    plt.imshow(mandelbort_set, cmap='hot_r',  vmin = vmin, vmax = vmax, extent= extent)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.colorbar()
    plt.show()

