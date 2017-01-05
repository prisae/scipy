from __future__ import division, print_function, absolute_import

from numpy import arange, log, log10
from numpy.fft.helper import fftshift, ifftshift, fftfreq
from bisect import bisect_left
from . import _fftpack

__all__ = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len',
           'rfftlogargs']


def rfftfreq(n, d=1.0):
    """DFT sample frequencies (for usage with rfft, irfft).

    The returned float array contains the frequency bins in
    cycles/unit (with zero at the start) given a window length `n` and a
    sample spacing `d`::

      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2]/(d*n)   if n is even
      f = [0,1,1,2,2,...,n/2-1,n/2-1,n/2,n/2]/(d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing. Default is 1.

    Returns
    -------
    out : ndarray
        The array of length `n`, containing the sample frequencies.

    Examples
    --------
    >>> from scipy import fftpack
    >>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> sig_fft = fftpack.rfft(sig)
    >>> n = sig_fft.size
    >>> timestep = 0.1
    >>> freq = fftpack.rfftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])

    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n = %s is not valid. "
                         "n must be a nonnegative integer." % n)

    return (arange(1, n + 1, dtype=int) // 2) / float(n * d)


def next_fast_len(target):
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    SciPy's FFTPACK has efficient functions for radix {2, 3, 4, 5}, so this
    returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first 5-smooth number greater than or equal to `target`.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    On a particular machine, an FFT of prime length takes 133 ms:

    >>> from scipy import fftpack
    >>> min_len = 10007  # prime length is worst case for speed
    >>> a = np.random.randn(min_len)
    >>> b = fftpack.fft(a)

    Zero-padding to the next 5-smooth length reduces computation time to
    211 us, a speedup of 630 times:

    >>> fftpack.helper.next_fast_len(min_len)
    10125
    >>> b = fftpack.fft(a, 10125)

    Rounding up to the next power of 2 is not optimal, taking 367 us to
    compute, 1.7 times as long as the 5-smooth size:

    >>> b = fftpack.fft(a, 16384)

    """
    hams = (8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128,
            135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250,
            256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450,
            480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729,
            750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125,
            1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536,
            1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160,
            2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
            3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
            3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000,
            5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400,
            6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
            8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000)

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= hams[-1]:
        return hams[bisect_left(hams, target)]

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            p2 = 2**((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def rfftlogargs(n, dlogr=0.01, logrc=0.0, mu=0.5, q=0, kr=1, kropt=0):
    """FFTLog input parameters (for usage with rfftl, irfftl).

    Return the required input points and the corresponding output points, the
    (adjusted) kr and the corresponding rk for `rfftl` and `irfftl`.

    Parameters
    ----------
    n : int
        Number of samples.

    dlogr : float, optional
        Separation between input-points (log10); may be positive or negative.
        Default is 0.01.

    logrc : float, optional
        Central point of periodic interval (log10). Default is 0.

    mu : float, optional
        Index of J_mu in Hankel transform; mu may be any real number, positive
        or negative. However, for `fftl` mu must be 0.5 for a sine transform,
        and -0.5 for a cosine transform. Only used if kropt is 1. Default is
        0.5.

    q : float, optional
        Exponent of power law bias; q may be any real number, positive or
        negative.  If in doubt, use q = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing.  Non-zero q may yield better approximations to the
        continuous Hankel transform for some functions. Only used if kropt is
        1. Default is 0 (unbiased).

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste). Default is 1.

    kropt : int, optional; {0, 1}
        - 0 to use input kr as is (default);
        - 1 to change kr to nearest low-ringing kr, quietly.


    Returns
    -------
    inppts : ndarray
        Array of length `n`, containing the sample input-points.
    outpts : ndarray
        Array of length `n`, containing the corresponding output-points.
    kr : float
        Low-ringing kr if kropt=1; else same as input.
    rk : float
        Inverse of kr, shifted if logrc != 0 and kr != 1.

    Examples
    --------
    >>> from scipy import fftpack
    >>> sig = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> sig_fft = fftpack.rfft(sig)
    >>> n = sig_fft.size
    >>> timestep = 0.1
    >>> outpts = fftpack.rfftfreq(n, d=timestep)
    >>> outpts
    array([ 0.  ,  1.25,  1.25,  2.5 ,  2.5 ,  3.75,  3.75,  5.  ])

    """

    # Central index (1/2 integral if n is even)
    nc = (n + 1)/2.0

    # Input points (frequencies)
    inppts = 10**(logrc + (arange(n)+1 - nc)*dlogr)

    # Get low-ringing kr
    if kropt == 1:
        kr = _fftpack.getkr(mu=mu, q=q, dlnr=dlogr*log(10.0), kr=kr,
                            kropt=kropt)

    # Central point log10(k_c) of periodic interval
    logkc = log10(kr) - logrc

    # rk = r_c/k_c
    rk = 10**(logrc - logkc)

    # Output points (times)
    outpts = 10**(logkc + (arange(n)+1 - nc)*dlogr)

    return inppts, outpts, kr, rk
