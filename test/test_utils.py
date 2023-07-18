import pytest
import mandelbrot.utils as utils
import numpy as np


class TestComplexArrayGenerator:
    def test_output(self):
        expected = np.array([[-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j],
                             [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
                             [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
                             [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
                             [-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j]])
        assert np.allclose(
            utils.generate_complex_array(real_range=(-2, 2), img_range=(-2, 2), size=5),
            expected
        )
