# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
from numpy.core.multiarray import normalize_axis_index
from . import _ni_support
from . import _nd_image

__all__ = ['fourier_gaussian', 'fourier_uniform', 'fourier_ellipsoid',
           'fourier_shift']


def _get_output_fourier(output, input, complex_only=False):
    types = [numpy.complex64, numpy.complex128]
    if not complex_only:
        types += [numpy.float32, numpy.float64]

    if output is None:
        if input.dtype in types:
            output = numpy.zeros(input.shape, dtype=input.dtype)
        else:
            output = numpy.zeros(input.shape, dtype=types[-1])
    elif type(output) is type:
        if output not in types:
            raise RuntimeError("output type not supported")
        output = numpy.zeros(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError("output shape not correct")
    return output


def _reshape_nd(arr, ndim, axis):
    """Promote a 1d array to ndim with non-singleton size along axis."""
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)


def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
    """
    Multidimensional Gaussian fourier filter.

    The array is multiplied with the fourier transform of a Gaussian
    kernel.

    Parameters
    ----------
    input : array_like
        The input array.
    sigma : float or sequence
        The sigma of the Gaussian kernel. If a float, `sigma` is the same for
        all axes. If a sequence, `sigma` has to contain one value for each
        axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_gaussian : ndarray
        The filtered input.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_gaussian(input_, sigma=4)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = numpy.asarray(input)
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = normalize_axis_index(axis, ndim)
    sigmas = _ni_support._normalize_sequence(sigma, ndim)

    output[...] = input
    for ax, (sigmak, ax_size) in enumerate(zip(sigmas, output.shape)):

        # compute the frequency grid in Hz
        if ax == axis and n > 0:
            arr = numpy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = numpy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)

        # compute the Gaussian weights
        arr *= arr
        scale = sigmak * sigmak / -2
        arr *= (4 * numpy.pi * numpy.pi) * scale
        numpy.exp(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output


def fourier_uniform(input, size, n=-1, axis=-1, output=None):
    """
    Multidimensional uniform fourier filter.

    The array is multiplied with the Fourier transform of a box of given
    size.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_uniform : ndarray
        The filtered input.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_uniform(input_, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = numpy.asarray(input)
    ndim = input.ndim
    output = _get_output_fourier(output, input)
    axis = normalize_axis_index(axis, ndim)
    sizes = _ni_support._normalize_sequence(size, ndim)

    output[...] = input
    for ax, (size, ax_size) in enumerate(zip(sizes, output.shape)):

        # compute the frequency grid in Hz
        if ax == axis and n > 0:
            arr = numpy.arange(ax_size, dtype=output.real.dtype)
            arr /= n
        else:
            arr = numpy.fft.fftfreq(ax_size)
        arr = arr.astype(output.real.dtype, copy=False)

        # compute the uniform filter weights
        arr *= size
        arr = numpy.sinc(arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output


def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
    """
    Multidimensional ellipsoid Fourier filter.

    The array is multiplied with the fourier transform of a ellipsoid of
    given sizes.

    Parameters
    ----------
    input : array_like
        The input array.
    size : float or sequence
        The size of the box used for filtering.
        If a float, `size` is the same for all axes. If a sequence, `size` has
        to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of filtering the input is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_ellipsoid : ndarray
        The filtered input.

    Notes
    -----
    This function is implemented for arrays of rank 1, 2, or 3.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import numpy.fft
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_ellipsoid(input_, size=20)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = numpy.asarray(input)
    output = _get_output_fourier(output, input)
    axis = normalize_axis_index(axis, input.ndim)
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    sizes = numpy.asarray(sizes, dtype=numpy.float64)
    if not sizes.flags.contiguous:
        sizes = sizes.copy()
    _nd_image.fourier_filter(input, sizes, n, axis, output, 2)
    return output


def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    """
    Multidimensional Fourier shift filter.

    The array is multiplied with the Fourier transform of a shift operation.

    Parameters
    ----------
    input : array_like
        The input array.
    shift : float or sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of shifting the input is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_shift : ndarray
        The shifted input.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import matplotlib.pyplot as plt
    >>> import numpy.fft
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = misc.ascent()
    >>> input_ = numpy.fft.fft2(ascent)
    >>> result = ndimage.fourier_shift(input_, shift=200)
    >>> result = numpy.fft.ifft2(result)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result.real)  # the imaginary part is an artifact
    >>> plt.show()
    """
    input = numpy.asarray(input)
    ndim = input.ndim
    output = _get_output_fourier(output, input, complex_only=True)
    axis = normalize_axis_index(axis, ndim)
    shifts = _ni_support._normalize_sequence(shift, ndim)

    output[...] = input
    for ax, (shiftk, ax_size) in enumerate(zip(shifts, output.shape)):
        if shiftk == 0:
            continue
        if ax == axis and n > 0:
            # cp.fft.rfftfreq(ax_size) * (-2j * numpy.pi * shiftk *  ax_size/n)
            arr = numpy.arange(ax_size, dtype=output.dtype)
            arr *= -2j * numpy.pi * shiftk / n
        else:
            arr = numpy.fft.fftfreq(ax_size)
            arr = arr * (-2j * numpy.pi * shiftk)
        numpy.exp(arr, out=arr)

        # reshape for broadcasting
        arr = _reshape_nd(arr, ndim=ndim, axis=ax)
        output *= arr

    return output
