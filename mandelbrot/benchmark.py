import time
import numpy as np 
import mandelbrot.mandelbrot as mb
import mandelbrot.utils as utils

def time_method(method_function, resolution=100, **kwargs) -> dict:
    """
    Times execution time of a mandelbrot method call

    Parameters
    -----
    method_call : str
        String representing method call
    resolution : int
        Resolution of the mandelbrot set (matrix with resolution x resolution elements)

    Returns
    -----
    method_call, execution_time: dict
        Dictionary with method call as key and execution time as value
    """
    X = utils.generate_complex_array(real_range=(-2, 1), img_range=(-1.5, 1.5), size=resolution)
    y = method_function(X, **kwargs)
    tic = time.time()
    y = method_function(X, **kwargs)
    toc = time.time() - tic
    return {method_function.__name__:toc, "result": y}

