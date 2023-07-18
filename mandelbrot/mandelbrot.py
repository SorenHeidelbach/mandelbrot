import numpy as np
from numba import jit, njit, prange


def mandelbrot_naive(c, max_iterations=100, divergence_limit=2):
    """
    Determines the divergence rate of the Mandelbrot equation for a complex number.

    This function determines the rate of divergence of a complex number in the Mandelbrot set, using a specified maximum number of iterations and divergence threshold.

    Parameters
    ----------
    c : complex
        The complex number to test.
    max_iterations : int
        The maximum number of iterations to use in determining the divergence of the Mandelbrot function.
    divergence_limit : float
        The threshold above which the Mandelbrot function is considered to have diverged.

    Returns
    -------
    int
        An integer representing the rate of divergence. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_naive(complex(0.5, 0.5), max_iterations=1000, divergence_limit=2.0)
    >>> print(divergence_rate)
    0.005
    """

    z =  0
    for i in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > divergence_limit:
            return (i + 1)/max_iterations
    return 1


@jit(nopython=True)
def mandelbrot_numba(c, max_iterations=100, divergence_limit=2):
    """
    Determines the divergence rate of the Mandelbrot equation for a complex number.

    This function determines the rate of divergence of a complex number in the Mandelbrot set, using a specified maximum number of iterations and divergence threshold.

    Parameters
    ----------
    c : complex
        The complex number to test.
    max_iterations : int
        The maximum number of iterations to use in determining the divergence of the Mandelbrot function.
    divergence_limit : float
        The threshold above which the Mandelbrot function is considered to have diverged.

    Returns
    -------
    int
        An integer representing the rate of divergence. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_numba(complex(0.5, 0.5), max_iterations=1000, divergence_limit=2.0)
    >>> print(divergence_rate)
    0.005
    """


    z =  0
    for i in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > divergence_limit:
            return (i + 1)/max_iterations
    return 1


def mandelbrot_set_naive(c):
    """
    Determines the divergence rate of the Mandelbrot equation for an array of complex numbers.

    This function determines the rate of divergence for each complex number in the array using a naive implementation with 'for loops'.

    Parameters
    ----------
    c : array
        An array of complex numbers to test.

    Returns
    -------
    convergence_rate : np.ndarray
        An array of integers representing the rate of divergence for each complex number. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_set_naive(np.array([[complex(0.5, 0.5), complex(-0.5, 0.5)], [complex(0.5, -0.5), complex(-0.5, -0.5)]]))
    >>> print(divergence_rate)
    [[0.05 1.  ]
     [0.05 1.  ]]
    """ 
    
    convergence_rate = np.empty_like(c, dtype=np.float64)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            convergence_rate[i, j] = mandelbrot_naive(c[i,j])
    return convergence_rate


@jit(nopython=True)
def mandelbrot_set_numba(c):
    """
    Determines the divergence rate of mandelbrot equation of a array of complex values.

    Naive implementation using 'for loops', but with numba decorator.

    Parameters
    ----------
    c : array
        An array of complex numbers to test.

    Returns
    -------
    convergence_rate : np.ndarray
        An array of integers representing the rate of divergence for each complex number. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_set_numba(np.array([[complex(0.5, 0.5), complex(-0.5, 0.5)], [complex(0.5, -0.5), complex(-0.5, -0.5)]]))
    >>> print(divergence_rate)
    [[0.05 1.  ]
     [0.05 1.  ]]
    """ 
    
    convergence_rate = np.empty_like(c, dtype="float")
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            convergence_rate[i, j] = mandelbrot_numba(c[i,j])
    return convergence_rate


@njit("Array(float64, 2, 'A', False, aligned=True)(Array(complex128, 2, 'A', False, aligned=True))", nopython=True, parallel=True)
def mandelbrot_set_numba_parallel(c):
    """
    Determines the divergence rate of mandelbrot equation of a array of complex values.

    Naive implementation using 'for loops', but with numba decorator in parallel mode.

    Parameters
    ----------
    c : array
        An array of complex numbers to test.

    Returns
    -------
    convergence_rate : np.ndarray
        An array of integers representing the rate of divergence for each complex number. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_set_numba_parallel(np.array([[complex(0.5, 0.5), complex(-0.5, 0.5)], [complex(0.5, -0.5), complex(-0.5, -0.5)]]))
    >>> print(divergence_rate)
    [[0.05 1.  ]
     [0.05 1.  ]]
    """ 
    
    convergence_rate = np.empty_like(c, dtype=np.float64)
    for i in range(c.shape[0]):
        for j in prange(c.shape[1]):
            convergence_rate[i, j] = mandelbrot_numba(c[i,j])
    return convergence_rate


@jit(nopython=True)
def mandelbrot_set_vectorised_numba(c, iterations=100, divergence_limit=2):
    """
    Determines the divergence rate of mandelbrot equation of a array of complex values

    Vectorised implementation using matrix operations. Numba decorator is used to speed up the process.

    Parameters
    ----------
    c : array
        An array of complex numbers to test.
    iterations : int
        The maximum number of iterations to use in determining the divergence of the Mandelbrot function.
    divergence_limit : float
        The threshold above which the Mandelbrot function is considered to have diverged.
        
    Returns
    -------
    convergence_rate : np.ndarray
        An array of integers representing the rate of divergence for each complex number. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_set_vectorised_numba(np.array([[complex(0.5, 0.5), complex(-0.5, 0.5)], [complex(0.5, -0.5), complex(-0.5, -0.5)]]))
    >>> print(divergence_rate)
    [[0.05 1.  ]
     [0.05 1.  ]]
    """ 
    
    z = np.zeros_like(c, dtype=np.complex128)
    divergence_rate = np.ones_like(c, dtype="float")
    mask = np.ones_like(c, dtype="bool")
    for i in range(iterations):
        z = z**2 + c
        z_abs = np.absolute(z)
        diverged = z_abs > divergence_limit
        divergence_rate = np.where(
            np.logical_and(diverged, mask), (i + 1)/iterations, divergence_rate
        )
        mask = np.invert(diverged)
    return divergence_rate


def mandelbrot_set_vectorised(c, iterations=100, divergence_limit=2):
    """
    Determines the divergence rate of mandelbrot equation of a array of complex values

    Vectorised implementation using matrix operations.

    Parameters
    ----------
    c : array
        An array of complex numbers to test.
    iterations : int
        The maximum number of iterations to use in determining the divergence of the Mandelbrot function.
    limit : float
        The threshold above which the Mandelbrot function is considered to have diverged.

    Returns
    -------
    convergence_rate : np.ndarray
        An array of integers representing the rate of divergence for each complex number. Convergence is represented by 1.

    Examples
    --------
    >>> divergence_rate = mandelbrot_set_vectorised(np.array([[complex(0.5, 0.5), complex(-0.5, 0.5)], [complex(0.5, -0.5), complex(-0.5, -0.5)]]))
    >>> print(divergence_rate)
    [[0.05 1.  ]
     [0.05 1.  ]]
    """ 
    z = np.zeros_like(c)
    divergence_rate = np.ones_like(c, dtype="float")
    mask = np.ones_like(c, dtype="bool")
    for i in range(iterations):
        z[mask] = z[mask]**2 + c[mask]
        z_abs = abs(z)
        diverged = z_abs > divergence_limit
        divergence_rate[diverged & mask] = (i + 1)/iterations
        mask = np.invert(diverged)
    return divergence_rate


if __name__ == "__main__":
    import pytest
    pytest.main(["-v"])