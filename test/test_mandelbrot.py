import pytest
import mandelbrot.utils as utils
import mandelbrot.mandelbrot as mb
import numpy as np


class TestMandelbrot:
    """
    Test different implementations of the Mandelbrot set return correct convergence values
    """
    X = np.array([[-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j],
               [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
               [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
               [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
               [-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j]])
    validation = np.array([[0.99, 0.99, 0.98, 0.99, 0.99],
                           [0.99, 0.97, 0.  , 0.98, 0.99],
                           [0.  , 0.  , 0.  , 0.97, 0.98],
                           [0.99, 0.97, 0.  , 0.98, 0.99],
                           [0.99, 0.99, 0.98, 0.99, 0.99]])
    def test_set_naive(self):
        assert np.allclose(
            mb.mandelbrot_set_naive(self.X),
            self.validation
        )
    def test_set_vectorised(self):
        assert np.allclose(
            mb.mandelbrot_set_vectorised(self.X),
            self.validation
        )
    def test_set_vectorised_numba(self):
        assert np.allclose(
            mb.mandelbrot_set_vectorised_numba(self.X),
            self.validation
        )
    def test_set_numba(self):
        assert np.allclose(
            mb.mandelbrot_set_numba(self.X),
            self.validation
        )
    def test_set_numba_parallel(self):
        assert np.allclose(
            mb.mandelbrot_set_numba_parallel(self.X),
            self.validation
        )


