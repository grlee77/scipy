import numpy as np

from .common import Benchmark

try:
    from scipy.ndimage import (fourier_gaussian, fourier_uniform,
                               fourier_shift, fourier_ellipsoid)
except ImportError:
    pass


class NdimageFourier(Benchmark):
    param_names = ['shape', 'dtype', 'real']
    params = [
        [(4096,), (64, 64), (2048, 2048), (16, 16, 16), (192, 192, 192)],
        [np.float32, np.float64],
        [False, True]
    ]

    def setup(self, shape, dtype, real):
        a = np.zeros(shape, dtype)
        sl0 = tuple([slice(0, 1)] * a.ndim)
        a[sl0] = 1.0
        if real:
            a = np.fft.rfft(a, shape[0], 0)
            for n in range(1, a.ndim):
                a = np.fft.fft(a, shape[n], n)
            self.args = (shape[0], 0)

        else:
            a = np.fft.fftn(a)
            self.args = ()
        cplx_type = np.promote_types(dtype, np.complex64)
        self.data = a.astype(cplx_type, copy=False)

    def time_fourier_gaussian(self, shape, dtype, real):
        fourier_gaussian(self.data, 2.5, *self.args)

    def time_fourier_uniform(self, shape, dtype, real):
        fourier_uniform(self.data, 10, *self.args)

    def time_fourier_shift(self, shape, dtype, real):
        fourier_shift(self.data, 1, *self.args)

    def time_fourier_ellipsoid(self, shape, dtype, real):
        fourier_ellipsoid(self.data, 5.0, *self.args)

    def peakmem_fourier_gaussian(self, shape, dtype, real):
        fourier_gaussian(self.data, 2.5, *self.args)

    def peakmem_fourier_ellipsoid(self, shape, dtype, real):
        fourier_ellipsoid(self.data, 5.0, *self.args)
