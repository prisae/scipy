"""
Discrete Fourier Transforms - basic.py
"""
# Created by Pearu Peterson, August,September 2002
from __future__ import division, print_function, absolute_import

__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
           'fft2','ifft2', 'rfftl']

from numpy import zeros, swapaxes
import numpy
from . import _fftpack

import atexit
atexit.register(_fftpack.destroy_zfft_cache)
atexit.register(_fftpack.destroy_zfftnd_cache)
atexit.register(_fftpack.destroy_drfft_cache)
atexit.register(_fftpack.destroy_cfft_cache)
atexit.register(_fftpack.destroy_cfftnd_cache)
atexit.register(_fftpack.destroy_rfft_cache)
atexit.register(_fftpack.destroy_fftlog_cache)
del atexit


def istype(arr, typeclass):
    return issubclass(arr.dtype.type, typeclass)


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)

    """
    if arr is original:
        return False
    if not isinstance(original, numpy.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None

# XXX: single precision FFTs partially disabled due to accuracy issues
#      for large prime-sized inputs.
#
#      See http://permalink.gmane.org/gmane.comp.python.scientific.devel/13834
#      ("fftpack test failures for 0.8.0b1", Ralf Gommers, 17 Jun 2010,
#       @ scipy-dev)
#
#      These should be re-enabled once the problems are resolved


def _is_safe_size(n):
    """
    Is the size of FFT such that FFTPACK can handle it in single precision
    with sufficient accuracy?

    Composite numbers of 2, 3, and 5 are accepted, as FFTPACK has those
    """
    n = int(n)

    if n == 0:
        return True

    # Divide by 3 until you can't, then by 5 until you can't
    for c in (3, 5):
        while n % c == 0:
            n //= c

    # Return True if the remainder is a power of 2
    return not n & (n-1)


def _fake_crfft(x, n, *a, **kw):
    if _is_safe_size(n):
        return _fftpack.crfft(x, n, *a, **kw)
    else:
        return _fftpack.zrfft(x, n, *a, **kw).astype(numpy.complex64)


def _fake_cfft(x, n, *a, **kw):
    if _is_safe_size(n):
        return _fftpack.cfft(x, n, *a, **kw)
    else:
        return _fftpack.zfft(x, n, *a, **kw).astype(numpy.complex64)


def _fake_rfft(x, n, *a, **kw):
    if _is_safe_size(n):
        return _fftpack.rfft(x, n, *a, **kw)
    else:
        return _fftpack.drfft(x, n, *a, **kw).astype(numpy.float32)


def _fake_cfftnd(x, shape, *a, **kw):
    if numpy.all(list(map(_is_safe_size, shape))):
        return _fftpack.cfftnd(x, shape, *a, **kw)
    else:
        return _fftpack.zfftnd(x, shape, *a, **kw).astype(numpy.complex64)

_DTYPE_TO_FFT = {
#        numpy.dtype(numpy.float32): _fftpack.crfft,
        numpy.dtype(numpy.float32): _fake_crfft,
        numpy.dtype(numpy.float64): _fftpack.zrfft,
#        numpy.dtype(numpy.complex64): _fftpack.cfft,
        numpy.dtype(numpy.complex64): _fake_cfft,
        numpy.dtype(numpy.complex128): _fftpack.zfft,
}

_DTYPE_TO_RFFT = {
#        numpy.dtype(numpy.float32): _fftpack.rfft,
        numpy.dtype(numpy.float32): _fake_rfft,
        numpy.dtype(numpy.float64): _fftpack.drfft,
}

_DTYPE_TO_FFTN = {
#        numpy.dtype(numpy.complex64): _fftpack.cfftnd,
        numpy.dtype(numpy.complex64): _fake_cfftnd,
        numpy.dtype(numpy.complex128): _fftpack.zfftnd,
#        numpy.dtype(numpy.float32): _fftpack.cfftnd,
        numpy.dtype(numpy.float32): _fake_cfftnd,
        numpy.dtype(numpy.float64): _fftpack.zfftnd,
}


def _asfarray(x):
    """Like numpy asfarray, except that it does not modify x dtype if x is
    already an array with a float dtype, and do not cast complex types to
    real."""
    if hasattr(x, "dtype") and x.dtype.char in numpy.typecodes["AllFloat"]:
        # 'dtype' attribute does not ensure that the
        # object is an ndarray (e.g. Series class
        # from the pandas library)
        return numpy.asarray(x, dtype=x.dtype)
    else:
        # We cannot use asfarray directly because it converts sequences of
        # complex to sequence of real
        ret = numpy.asarray(x)
        if ret.dtype.char not in numpy.typecodes["AllFloat"]:
            return numpy.asfarray(x)
        return ret


def _fix_shape(x, n, axis):
    """ Internal auxiliary function for _raw_fft, _raw_fftnd."""
    s = list(x.shape)
    if s[axis] > n:
        index = [slice(None)]*len(s)
        index[axis] = slice(0,n)
        x = x[index]
        return x, False
    else:
        index = [slice(None)]*len(s)
        index[axis] = slice(0,s[axis])
        s[axis] = n
        z = zeros(s,x.dtype.char)
        z[index] = x
        return z, True


def _raw_fft(x, n, axis, direction, overwrite_x, work_function):
    """ Internal auxiliary function for fft, ifft, rfft, irfft."""
    if n is None:
        n = x.shape[axis]
    elif n != x.shape[axis]:
        x, copy_made = _fix_shape(x,n,axis)
        overwrite_x = overwrite_x or copy_made

    if n < 1:
        raise ValueError("Invalid number of FFT data points "
                         "(%d) specified." % n)

    if axis == -1 or axis == len(x.shape)-1:
        r = work_function(x,n,direction,overwrite_x=overwrite_x)
    else:
        x = swapaxes(x, axis, -1)
        r = work_function(x,n,direction,overwrite_x=overwrite_x)
        r = swapaxes(r, axis, -1)
    return r


def fft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

    ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

    Parameters
    ----------
    x : array_like
        Array to Fourier transform.
    n : int, optional
        Length of the Fourier transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the fft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    z : complex ndarray
        with the elements::

            [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
            [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

        where::

            y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

        Note that ``y(-j) = y(n-j).conjugate()``.

    See Also
    --------
    ifft : Inverse FFT
    rfft : FFT of a real sequence

    Notes
    -----
    The packing of the result is "standard": If ``A = fft(a, n)``, then
    ``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
    positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
    terms, in order of decreasingly negative frequency. So for an 8-point
    transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
    To rearrange the fft output so that the zero-frequency component is
    centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

    For `n` even, ``A[n/2]`` contains the sum of the positive and
    negative-frequency terms.  For `n` even and `x` real, ``A[n/2]`` will
    always be real.

    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    If the data type of `x` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time.  To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum.  If the data is both real and
    symmetrical, the `dct` can again double the efficiency, by generating
    half of the spectrum from half of the signal.

    Examples
    --------
    >>> from scipy.fftpack import fft, ifft
    >>> x = np.arange(5)
    >>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
    True

    """
    tmp = _asfarray(x)

    try:
        work_function = _DTYPE_TO_FFT[tmp.dtype]
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
        overwrite_x = 1

    overwrite_x = overwrite_x or _datacopied(tmp, x)

    if n is None:
        n = tmp.shape[axis]
    elif n != tmp.shape[axis]:
        tmp, copy_made = _fix_shape(tmp,n,axis)
        overwrite_x = overwrite_x or copy_made

    if n < 1:
        raise ValueError("Invalid number of FFT data points "
                         "(%d) specified." % n)

    if axis == -1 or axis == len(tmp.shape) - 1:
        return work_function(tmp,n,1,0,overwrite_x)

    tmp = swapaxes(tmp, axis, -1)
    tmp = work_function(tmp,n,1,0,overwrite_x)
    return swapaxes(tmp, axis, -1)


def ifft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete inverse Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)`` where

    ``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,
        `x` is truncated.  If ``n > x.shape[axis]``, `x` is zero-padded.
        The default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the ifft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    ifft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    fft : Forward FFT

    Notes
    -----
    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    If the data type of `x` is real, a "real IFFT" algorithm is automatically
    used, which roughly halves the computation time.

    """
    tmp = _asfarray(x)

    try:
        work_function = _DTYPE_TO_FFT[tmp.dtype]
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
        overwrite_x = 1

    overwrite_x = overwrite_x or _datacopied(tmp, x)

    if n is None:
        n = tmp.shape[axis]
    elif n != tmp.shape[axis]:
        tmp, copy_made = _fix_shape(tmp,n,axis)
        overwrite_x = overwrite_x or copy_made

    if n < 1:
        raise ValueError("Invalid number of FFT data points "
                         "(%d) specified." % n)

    if axis == -1 or axis == len(tmp.shape) - 1:
        return work_function(tmp,n,-1,1,overwrite_x)

    tmp = swapaxes(tmp, axis, -1)
    tmp = work_function(tmp,n,-1,1,overwrite_x)
    return swapaxes(tmp, axis, -1)


def rfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Discrete Fourier transform of a real sequence.

    Parameters
    ----------
    x : array_like, real-valued
        The data to transform.
    n : int, optional
        Defines the length of the Fourier transform.  If `n` is not specified
        (the default) then ``n = x.shape[axis]``.  If ``n < x.shape[axis]``,
        `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.
    axis : int, optional
        The axis along which the transform is applied.  The default is the
        last axis.
    overwrite_x : bool, optional
        If set to true, the contents of `x` can be overwritten. Default is
        False.

    Returns
    -------
    z : real ndarray
        The returned real array contains::

          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even
          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd

        where::

          y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
          j = 0..n-1

        Note that ``y(-j) == y(n-j).conjugate()``.

    See Also
    --------
    fft, irfft, scipy.fftpack.basic

    Notes
    -----
    Within numerical accuracy, ``y == rfft(irfft(y))``.

    Examples
    --------
    >>> from scipy.fftpack import fft, rfft
    >>> a = [9, -9, 1, 3]
    >>> fft(a)
    array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])
    >>> rfft(a)
    array([  4.,   8.,  12.,  16.])

    """
    tmp = _asfarray(x)

    if not numpy.isrealobj(tmp):
        raise TypeError("1st argument must be real sequence")

    try:
        work_function = _DTYPE_TO_RFFT[tmp.dtype]
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    overwrite_x = overwrite_x or _datacopied(tmp, x)

    return _raw_fft(tmp,n,axis,1,overwrite_x,work_function)


def irfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return inverse discrete Fourier transform of real sequence x.

    The contents of `x` are interpreted as the output of the `rfft`
    function.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.
        If n < x.shape[axis], x is truncated.
        If n > x.shape[axis], x is zero-padded.
        The default results in n = x.shape[axis].
    axis : int, optional
        Axis along which the ifft's are computed; the default is over
        the last axis (i.e., axis=-1).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    irfft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    rfft, ifft

    Notes
    -----
    The returned real array contains::

        [y(0),y(1),...,y(n-1)]

    where for n is even::

        y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0] + (-1)**(j) x[n-1])

    and for n is odd::

        y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0])

    c.c. denotes complex conjugate of preceding expression.

    For details on input parameters, see `rfft`.

    """
    tmp = _asfarray(x)
    if not numpy.isrealobj(tmp):
        raise TypeError("1st argument must be real sequence")

    try:
        work_function = _DTYPE_TO_RFFT[tmp.dtype]
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    overwrite_x = overwrite_x or _datacopied(tmp, x)

    return _raw_fft(tmp,n,axis,-1,overwrite_x,work_function)


def _raw_fftnd(x, s, axes, direction, overwrite_x, work_function):
    """ Internal auxiliary function for fftnd, ifftnd."""
    if s is None:
        if axes is None:
            s = x.shape
        else:
            s = numpy.take(x.shape, axes)

    s = tuple(s)
    if axes is None:
        noaxes = True
        axes = list(range(-x.ndim, 0))
    else:
        noaxes = False
    if len(axes) != len(s):
        raise ValueError("when given, axes and shape arguments "
                         "have to be of the same length")

    for dim in s:
        if dim < 1:
            raise ValueError("Invalid number of FFT data points "
                             "(%s) specified." % (s,))

    # No need to swap axes, array is in C order
    if noaxes:
        for i in axes:
            x, copy_made = _fix_shape(x, s[i], i)
            overwrite_x = overwrite_x or copy_made
        return work_function(x,s,direction,overwrite_x=overwrite_x)

    # We ordered axes, because the code below to push axes at the end of the
    # array assumes axes argument is in ascending order.
    id = numpy.argsort(axes)
    axes = [axes[i] for i in id]
    s = [s[i] for i in id]

    # Swap the request axes, last first (i.e. First swap the axis which ends up
    # at -1, then at -2, etc...), such as the request axes on which the
    # operation is carried become the last ones
    for i in range(1, len(axes)+1):
        x = numpy.swapaxes(x, axes[-i], -i)

    # We can now operate on the axes waxes, the p last axes (p = len(axes)), by
    # fixing the shape of the input array to 1 for any axis the fft is not
    # carried upon.
    waxes = list(range(x.ndim - len(axes), x.ndim))
    shape = numpy.ones(x.ndim)
    shape[waxes] = s

    for i in range(len(waxes)):
        x, copy_made = _fix_shape(x, s[i], waxes[i])
        overwrite_x = overwrite_x or copy_made

    r = work_function(x, shape, direction, overwrite_x=overwrite_x)

    # reswap in the reverse order (first axis first, etc...) to get original
    # order
    for i in range(len(axes), 0, -1):
        r = numpy.swapaxes(r, -i, axes[-i])

    return r


def fftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return multidimensional discrete Fourier transform.

    The returned array contains::

      y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

    where d = len(x.shape) and n = x.shape.
    Note that ``y[..., -j_i, ...] = y[..., n_i-j_i, ...].conjugate()``.

    Parameters
    ----------
    x : array_like
        The (n-dimensional) array to transform.
    shape : tuple of ints, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
        length ``shape[i]``.
    axes : array_like of ints, optional
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed.  Default is False.

    Returns
    -------
    y : complex-valued n-dimensional numpy array
        The (n-dimensional) DFT of the input array.

    See Also
    --------
    ifftn

    Examples
    --------
    >>> from scipy.fftpack import fftn, ifftn
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, fftn(ifftn(y)))
    True

    """
    return _raw_fftn_dispatch(x, shape, axes, overwrite_x, 1)


def _raw_fftn_dispatch(x, shape, axes, overwrite_x, direction):
    tmp = _asfarray(x)

    try:
        work_function = _DTYPE_TO_FFTN[tmp.dtype]
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
        overwrite_x = 1

    overwrite_x = overwrite_x or _datacopied(tmp, x)
    return _raw_fftnd(tmp,shape,axes,direction,overwrite_x,work_function)


def ifftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return inverse multi-dimensional discrete Fourier transform of
    arbitrary type sequence x.

    The returned array contains::

      y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)

    where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.

    For description of parameters see `fftn`.

    See Also
    --------
    fftn : for detailed information.

    """
    return _raw_fftn_dispatch(x, shape, axes, overwrite_x, -1)


def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    """
    2-D discrete Fourier transform.

    Return the two-dimensional discrete Fourier transform of the 2-D argument
    `x`.

    See Also
    --------
    fftn : for detailed information.

    """
    return fftn(x,shape,axes,overwrite_x)


def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    """
    2-D discrete inverse Fourier transform of real or complex sequence.

    Return inverse two-dimensional discrete Fourier transform of
    arbitrary type sequence x.

    See `ifft` for more information.

    See also
    --------
    fft2, ifft

    """
    return ifftn(x,shape,axes,overwrite_x)


def _raw_fftlog(x, n, mu, q, dlogr, kr, rk, axis, direction, overwrite_x,
                work_function):
    """ Internal auxiliary function for rfftl."""

    # TODO: in fftlog, n must be equals x.size; is the implementation of
    # n, axis, and overwrite_x correct?
    if n is None:
        n = x.shape[axis]
    elif n != x.shape[axis]:
        x, copy_made = _fix_shape(x, n, axis)
        overwrite_x = overwrite_x or copy_made

    if n < 1:
        raise ValueError("Invalid number of FFT data points "
                         "(%d) specified." % n)

    testax = (axis == -1 or axis == len(x.shape)-1)
    if not testax:
        x = swapaxes(x, axis, -1)

    r = work_function(x, n, mu, q, dlogr*numpy.log(10.0), kr, rk, direction,
                      overwrite_x)

    if not testax:
        r = swapaxes(r, axis, -1)

    return r


def rfftl(x, dlogr, mu=0.5, q=0.0, kr=1.0, rk=1.0, n=None, axis=-1,
          overwrite_x=False):
    """Fourier transform of a logarithmically spaced periodic sequence.

    Fast Fourier transform of a real, discrete periodic sequence of
    logarithmically spaced points.

    `rfftl` computes a discrete version of the Fourier sine (if mu = 1/2) or
    cosine (if mu = -1/2) transform

    .. math::

        \tilde{A} = \sqrt{2/\pi} \int_0^\infty A(r) \sin(kr) dr,

        \tilde{A} = \sqrt{2/\pi} \int_0^\infty A(r) \cos(kr) dr


    by making the substitutions

    .. math::

        A(r) = a(r) r^{\mu-1/2},

        \tilde{A}(k) = \tilde{a}(k) k^{-\mu -1/2}

    and applying a biased Hankel transform to a(r).

    A call to `rfftl` followed by a call to `irfftl` (and rk unchanged), or
    vice versa, leaves the array a unchanged.


    Parameters
    ----------
    x : array_like, real-valued
        Array A(r) to transform: a(j) is A(r_j) at r_j = r_c exp[(j-jc) dlnr],
        where jc = (n+1)/2 = central index of array.

    dlogr : float, optional
        Separation between input-points (log10); may be positive or negative.
        Default is 0.01.

    mu : float, optional; {-0.5, 0.5}
        Index of J_mu in Hankel transform: 0.5 for a sine transform and -0.5
        for a cosine transform. Default is 0.5.

    q : float, optional
        Exponent of power law bias; q may be any real number, positive or
        negative.  If in doubt, use q = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing.  Non-zero q may yield better approximations to the
        continuous Hankel transform for some functions. Default is 0
        (unbiased).

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste). Default is 1.

    rk : float, optional
        r_c/k_c = r_j/k_j (a constant, the same constant for any j); rk is not
        (necessarily) the same quantity as kr.  rk is used only to multiply the
        output array by sqrt(rk)^dir, so if you want to do the normalization
        later, or you don't care about the normalization, you can set rk = 1.
        Default is 1.

    n : int, optional
        Defines the length of the Fourier transform.  If `n` is not specified
        (the default) then ``n = x.shape[axis]``.  If ``n < x.shape[axis]``,
        `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.

    axis : int, optional
        The axis along which the transform is applied.  The default is the
        last axis.

    overwrite_x : bool, optional
        If set to true, the contents of `x` can be overwritten. Default is
        False.

    Returns
    -------
    y : real ndarray
        Transformed array Ã(k): a(j) is Ã(k_j) at k_j = k_c exp[(j-jc) dlnr].

    See Also
    --------
    irfftl, scipy.fftpack.basic

    References
    ----------
    .. [1] 'Uncorrelated modes of the non-linear power spectrum', by A. J. S.
           Hamilton, `Monthly Notices of the Royal Astronomical Society` vol.
           312, pp. 257-284, http://dx.doy.org/10.1046/j.1365-8711.2000.03071.x
           (2000). Website of FFTLog: http://casa.colorado.edu/~ajsh/FFTLog.

    Examples
    --------
    >>> from scipy.fftpack import rfftlogargs, rfftl
    >>> # Get fftl-arguments
    >>> n, dlogr, logrc = 4, .1, 0
    >>> q = 0
    >>> mu = 0.5
    >>> w, t, kr, rk = rfftlogargs(n, dlogr, logrc, mu, q, 1, 1)
    >>> rk /= 2/np.pi    # Scale
    >>> # Analytical solution
    >>> fw = np.sqrt(np.pi/2/w)  # Frequency domain
    >>> ft = 1/np.sqrt(t)        # Time domain
    >>> # FFTLog
    >>> fftl = rfftl(fw, dlogr=dlogr, mu=0.5, q=q, kr=kr, rk=rk)
    >>> fftl *= 2/np.pi  # Scale back
    >>> # Print result
    >>> print('Input      :', fw)
    >>> print('Analytical :', ft)
    >>> print('fftl       :', fftl)
    Input      : [ 1.48956664  1.32757767  1.18320484  1.05453243]
    Analytical : [ 1.15380264  1.02832769  0.91649802  0.81682972]
    fftl       : [ 1.15380264  1.02832769  0.91649802  0.81682972]

    """

    # Check that mu is {0.5, -0.55}
    if mu not in [0.5, -0.5]:
        raise ValueError("mu must be either 0.5 (sine) or -0.5 (cosine)")

    tmp = _asfarray(x)

    if not numpy.isrealobj(tmp):
        raise TypeError("1st argument must be real sequence")

    try:  # TODO: Check types, which ones are supported?
        # work_function = _DTYPE_TO_RFFT[tmp.dtype]
        work_function = _fftpack.drfftl
    except KeyError:
        raise ValueError("type %s is not supported" % tmp.dtype)

    # TODO: Is the implementation of overwrite_x correct?
    overwrite_x = overwrite_x or _datacopied(tmp, x)

    y = _raw_fftlog(tmp, n, mu, q, dlogr, kr, rk, axis, 1, overwrite_x,
                    work_function)

    return y
