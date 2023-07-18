


import time
import numpy as np 
import mandelbrot.mandelbrot as mb
import mandelbrot.utils as utils

def time_method(method_function, resolution=100) -> dict:
    """
    Times execution time of a mandelbrot method call

    INPUT::
    method_call:    String representing method call
    resolution:     Resolution of the mandelbrot set (matrix with resolution x resolution elements)

    OUTPUT::
    method_call, execution_time:    String representing method call
    """
    X = utils.generate_complex_array(real_range=(-2, 1), img_range=(-1.5, 1.5), size=resolution)
    y = method_function(X)
    tic = time.time()
    y = method_function(X)
    toc = time.time() - tic
    return {method_function.__name__:toc}

