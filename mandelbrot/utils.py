import numpy as np

def generate_complex_array(real_range: tuple[float, float], img_range: tuple[float, float], size: int) -> np.ndarray:
    """
    Generates a complex array of size x size elements with real and imaginary values
    """
    imaginary_vals = 1j * np.linspace(img_range[0], img_range[1], size)
    real_vals = np.linspace(real_range[0], real_range[1], size)
    complex_vals = np.add.outer(imaginary_vals, real_vals, dtype=np.complex128)
    return complex_vals