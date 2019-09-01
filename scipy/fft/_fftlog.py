"""
Python version of the FFTLog algorithm
======================================

This is a Python version of the Fortran-FFTLog algorithm by Andrew Hamilton:

Hamilton, A. J. S., 2000, Uncorrelated modes of the non-linear power spectrum:
Monthly Notices of the Royal Astronomical Society, 312, pages 257-284; DOI:
10.1046/j.1365-8711.2000.03071.x; Website of FFTLog:
http://casa.colorado.edu/~ajsh/FFTLog.

FFTLog computes the discrete Fast Fourier Transform or Fast Hankel Transform
(of arbitrary real index) of a periodic logarithmic sequence.

The function `scipy.special.loggamma` replaces the file `cdgamma.f` in the
original code, and the functions `rfft` and `irfft` from `scipy.fftpack`
replace the files `drffti.f`, `drfftf.f`, and `drfftb.f` in the original code.

Permission to distribute this modified FFTLog under the BSD-3-Clause license
has been granted (email from Andrew Hamilton to Dieter Werthmüller dated 07
October 2016).

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.special import loggamma
from ._basic import rfft, irfft

__all__ = ['fftlog', 'fftlogargs']


def fftlog(x, spacing, transform='sine', bias=0.0, kr=1.0, rk=1.0):
    """Fourier transform of a logarithmically spaced periodic sequence.

    Fast Fourier transform of a real, discrete periodic sequence of
    logarithmically spaced points.

    `fftlog` computes a discrete version of the Fourier sine or cosine
    transform

    .. math::

        G = \sqrt{2/\pi} \int_0^\infty F(r) \sin(kr) dr,

        G = \sqrt{2/\pi} \int_0^\infty F(r) \cos(kr) dr


    by making the substitutions

    .. math::

        F(r) = f(r) r^{ \mu - 1/2},

        G(k) = g(k) k^{-\mu - 1/2}

    and applying a biased Hankel transform to f(r);
    mu = 1/2 for the sine and -1/2 for the cosine transform.

    Parameters
    ----------
    x : array_like, real-valued
        Array F(r) to transform: f(j) is F(r_j) at r_j = r_c exp[(j-jc) dlnr],
        where jc = (n+1)/2 = central index of array.

    spacing : float, optional
        Separation between input-points (log10); may be positive or negative.
        Default is 0.01.

    transform : string, optional; {'sine', 'cosine'}
        Transform type to use, which defines index of J_mu in Hankel transform:
        mu is 0.5 for a sine transform and -0.5 for a cosine transform. Default
        is 'sine' (mu=0.5).

    bias : float, optional
        Exponent of power law bias; bias may be any real number, positive or
        negative. If in doubt, use bias = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing. Non-zero bias may yield better approximations to the
        continuous Hankel transform for some functions. Default is 0
        (unbiased).

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste). Default is 1.

    rk : float, optional
        r_c/k_c = r_j/k_j (a constant, the same constant for any j); rk is not
        (necessarily) the same quantity as kr. rk is used only to multiply the
        output array by sqrt(rk)^dir, so if you want to do the normalization
        later, or you don't care about the normalization, you can set rk = 1.
        Default is 1.

    Returns
    -------
    y : real ndarray
        Transformed array G(k): g(j) is G(k_j) at k_j = k_c exp[(j-jc) dlnr].

    .. versionadded:: 1.?.0

    References
    ----------
    .. [1] 'Uncorrelated modes of the non-linear power spectrum', by A. J. S.
           Hamilton, `Monthly Notices of the Royal Astronomical Society` vol.
           312, pp. 257-284, http://dx.doy.org/10.1046/j.1365-8711.2000.03071.x
           (2000). Website of FFTLog: http://casa.colorado.edu/~ajsh/FFTLog.

    Examples
    --------
    >>> from scipy.fft import fftlog, fftlogargs

    Get fftlog-arguments

    >>> n, spacing, center = 4, .1, 0
    >>> bias = 0
    >>> transform = 'sine'
    >>> w, t, kr, rk = fftlogargs(n, spacing, center, transform, bias, 1, 1)
    >>> rk /= 2/np.pi    # Scale

    Analytical solution

    >>> fw = np.sqrt(np.pi/2/w)  # Frequency domain
    >>> ft = 1/np.sqrt(t)        # Time domain

    FFTLog

    >>> fftl = fftlog(fw, spacing, transform, bias, kr, rk)
    >>> fftl *= 2/np.pi  # Scale back

    Print result

    >>> print('Input      :', fw)
    Input      : [ 1.48956664  1.32757767  1.18320484  1.05453243]
    >>> print('Analytical :', ft)
    Analytical : [ 1.15380264  1.02832769  0.91649802  0.81682972]
    >>> print('fftlog     :', fftl)
    fftlog     : [ 1.15380264  1.02832769  0.91649802  0.81682972]

    """

    # Check that transform is {'sine', or 'cosine'}
    if transform not in ['sine', 'cosine']:
        raise ValueError("transform must be either 'sine' or 'cosine'.")
    if transform == 'sine':
        mu = 0.5
    else:
        mu = -0.5

    tmp = _asfarray(x)

    if len(tmp) < 1:
        raise ValueError("Invalid number of FFT data points "
                         "(%d) specified." % len(tmp))

    dlnr = spacing*np.log(10.0)
    return fftl(tmp, mu, bias, dlnr, kr, rk, 1)


def fftlogargs(n, spacing=0.01, center=0.0, transform='sine', bias=0, kr=1,
               kropt=False):
    """FFTLog input parameters (for usage with fftlog).

    Return the required input points and the corresponding output points, the
    (adjusted) kr and the corresponding rk for `fftlog`.

    Parameters
    ----------
    n : int
        Number of samples.

    spacing : float, optional
        Separation between input-points (log10); may be positive or negative.
        Default is 0.01.

    center : float, optional
        Central point of periodic interval (log10). Default is 0.

    transform : string, optional; {'sine', 'cosine'}
        Transform type to use, which defines index of J_mu in Hankel transform:
        mu is 0.5 for a sine transform and -0.5 for a cosine transform. Default
        is 'sine' (mu=0.5).

    bias : float, optional
        Exponent of power law bias; bias may be any real number, positive or
        negative. If in doubt, use bias = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing. Non-zero bias may yield better approximations to the
        continuous Hankel transform for some functions. Only used if kropt is
        True. Default is 0 (unbiased).

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste). Default is 1.

    kropt : bool, optional
        - False to use input kr as is (default);
        - True to change kr to nearest low-ringing value.


    Returns
    -------
    inppts : ndarray
        Array of length `n`, containing the sample input-points.
    outpts : ndarray
        Array of length `n`, containing the corresponding output-points.
    kr : float
        Low-ringing kr if kropt=True; else same as input.
    rk : float
        Inverse of kr, shifted if center != 0 and kr != 1.

    .. versionadded:: 1.0.0

    Examples
    --------
    >>> from scipy.fftpack import fftlogargs
    >>> intpts, outpts, kr, rk = fftlogargs(n=4, kr=1, kropt=True)
    >>> print('intpts :', intpts)
    intpts : [ 0.96605088  0.98855309  1.01157945  1.03514217]
    >>> print('outpts :', outpts)
    outpts : [ 0.97306236  0.9957279   1.01892138  1.04265511]
    >>> print('kr     :', kr)
    kr     : 1.0072578812188107
    >>> print('rk     :', rk)
    rk     : 0.992794416054

    """
    # Check that transform is {'sine', or 'cosine'}; get mu
    if transform not in ['sine', 'cosine']:
        raise ValueError("transform must be either 'sine' or 'cosine'.")
    if transform == 'sine':
        mu = 0.5
    else:
        mu = -0.5

    # Central index (1/2 integral if n is even)
    nc = (n + 1)/2.0

    # Input points (frequencies)
    inppts = 10**(center + (np.arange(n)+1 - nc)*spacing)

    # Get low-ringing kr
    if kropt:
        dlnr = spacing*np.log(10.0)
        kr = krgood(mu=mu, q=bias, dlnr=dlnr, kr=kr)

    # Central point log10(k_c) of periodic interval
    logkc = np.log10(kr) - center

    # rk = r_c/k_c
    rk = 10**(center - logkc)

    # Output points (times)
    outpts = 10**(logkc + (np.arange(n)+1 - nc)*spacing)

    return inppts, outpts, kr, rk


def fhti(n, mu, dlnr, q=0, kr=1, kropt=0):
    """Initialize the working array xsave used by fftlog, fht, and fhtq.

    fhti initializes the working array xsave used by fftlog, fht, and fhtq.
    fhti need be called once, whereafter fftlog, fht, or fhtq may be called
    many times, as long as n, mu, q, dlnr, and kr remain unchanged. fhti should
    be called each time n, mu, q, dlnr, or kr is changed. The work array xsave
    should not be changed between calls to fftlog, fht, or fhtq.

    Parameters
    ----------
    n : int
        Number of points in the array to be transformed; n may be any positive
        integer, but the FFT routines run fastest if n is a product of small
        primes 2, 3, 5.

    mu : float
        Index of J_mu in Hankel transform; mu may be any real number, positive
        or negative.

    dlnr : float
        Separation between natural log of points; dlnr may be positive or
        negative.

    q : float, optional
        Exponent of power law bias; q may be any real number, positive or
        negative.  If in doubt, use q = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing.  Non-zero q may yield better approximations to the
        continuous Hankel transform for some functions.
        Defaults to 0 (unbiased).

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste).

    kropt : int, optional; {0, 1, 2, 3}
        - 0 to use input kr as is (default);
        - 1 to change kr to nearest low-ringing kr, quietly;
        - 2 to change kr to nearest low-ringing kr, verbosely;
        - 3 for option to change kr interactively.

    Returns
    -------
    kr : float, optional
        kr, adjusted depending on kropt.

    xsave : array
        Working array used by fftlog, fht, and fhtq. Dimension:
        - for q = 0 (unbiased transform): n+3
        - for q != 0 (biased transform): 1.5*n+4
        If odd, last element is not needed.

    """

    # adjust kr
    if kropt == 0:    # keep kr as is
        pass
    elif kropt == 1:  # change kr to low-ringing kr quietly
        kr = krgood(mu, q, dlnr, kr)
    elif kropt == 2:  # change kr to low-ringing kr verbosely
        d = krgood(mu, q, dlnr, kr)
        if abs(kr/d - 1) >= 1e-15:
            kr = d
            print(" kr changed to ", kr)
    else:             # option to change kr to low-ringing kr interactively
        d = krgood(mu, q, dlnr, kr)
        if abs(kr/d-1.0) >= 1e-15:
            print(" change kr = ", kr)
            print(" to low-ringing kr = ", d)
            go = input("? [CR, y=yes, n=no, x=exit]: ")
            if go.lower() in ['', 'y']:
                kr = d
                print(" kr changed to ", kr)
            elif go.lower() == 'n':
                print(" kr left unchanged at ", kr)
            else:
                print("exit")
                return False

    # return if n is <= 0
    if n <= 0:
        return kr

    # The normal FFT is not initialized here as in the original FFTLog code, as
    # the `scipy.fftpack`-FFT routines `rfft` and `irfft` do that internally.
    # Therefore xsave in `pyfftlog` is 2*n+15 elements shorter, and named
    # xsave to not confuse it with xsave from the FFT.

    if q == 0:  # unbiased case (q = 0)
        ln2kr = np.log(2.0/kr)
        xp = (mu + 1)/2.0
        d = np.pi/(n*dlnr)

        m = np.arange(1, (n+1)/2)
        y = m*d  # y = m*pi/(n*dlnr)
        zp = loggamma(xp + 1j*y)
        arg = 2.0*(ln2kr*y + zp.imag)  # Argument of kr^(-2 i y) U_mu(2 i y)

        # Arange xsave: [q, dlnr, kr, cos, sin]
        xsave = np.empty(2*arg.size+3)
        xsave[0] = q
        xsave[1] = dlnr
        xsave[2] = kr
        xsave[3::2] = np.cos(arg)
        xsave[4::2] = np.sin(arg)

        # Altogether 3 + 2*(n/2) elements used for q = 0, which is n+3 for even
        # n, n+2 for odd n.

    else:       # biased case (q != 0)
        ln2 = np.log(2.0)
        ln2kr = np.log(2.0/kr)
        xp = (mu + 1 + q)/2.0
        xm = (mu + 1 - q)/2.0

        # first element of rest of xsave
        y = 0

        # case where xp or xm is a negative integer
        xpnegi = np.round(xp) == xp and xp <= 0
        xmnegi = np.round(xm) == xm and xm <= 0
        if xpnegi or xmnegi:

            # case where xp and xm are both negative integers
            # U_mu(q) = 2^q Gamma[xp]/Gamma[xm] is finite in this case
            if xpnegi and xmnegi:
                # Amplitude and Argument of U_mu(q)
                amp = np.exp(ln2*q)
                if xp > xm:
                    m = np.arange(1,  np.round(xp - xm)+1)
                    amp *= xm + m - 1
                elif xp < xm:
                    m = np.arange(1,  np.round(xm - xp)+1)
                    amp /= xp + m - 1
                arg = np.round(xp + xm)*np.pi

            else:  # one of xp or xm is a negative integer
                # Transformation is singular if xp is -ve integer, and inverse
                # transformation is singular if xm is -ve integer, but
                # transformation may be well-defined if sum_j a_j = 0, as may
                # well occur in physical cases.  Policy is to drop the
                # potentially infinite constant in the transform.

                if xpnegi:
                    print('fhti: (mu+1+q)/2 =', np.round(xp), 'is -ve integer',
                          ', yields singular transform:\ntransform will omit',
                          'additive constant that is generically infinite,',
                          '\nbut that may be finite or zero if the sum of the',
                          'elements of the input array a_j is zero.')
                else:
                    print('fhti: (mu+1-q)/2 =', np.round(xm), 'is -ve integer',
                          ', yields singular inverse transform:\n inverse',
                          'transform will omit additive constant that is',
                          'generically infinite,\nbut that may be finite or',
                          'zero if the sum of the elements of the input array',
                          'a_j is zero.')
                amp = 0
                arg = 0

        else:  # neither xp nor xm is a negative integer
            zp = loggamma(xp + 1j*y)
            zm = loggamma(xm + 1j*y)

            # Amplitude and Argument of U_mu(q)
            amp = np.exp(ln2*q + zp.real - zm.real)
            # note +Im(zm) to get conjugate value below real axis
            arg = zp.imag + zm.imag

        # first element: cos(arg) = ±1, sin(arg) = 0
        xsave1 = amp*np.cos(arg)

        # remaining elements of xsave
        d = np.pi/(n*dlnr)
        m = np.arange(1, (n+1)/2)
        y = m*d  # y = m pi/(n dlnr)
        zp = loggamma(xp + 1j*y)
        zm = loggamma(xm + 1j*y)
        # Amplitude and Argument of kr^(-2 i y) U_mu(q + 2 i y)
        amp = np.exp(ln2*q + zp.real - zm.real)
        arg = 2*ln2kr*y + zp.imag + zm.imag

        # Arrange xsave: [q, dlnr, kr, xsave1, cos, sin]
        xsave = np.empty(3*arg.size+4)
        xsave[0] = q
        xsave[1] = dlnr
        xsave[2] = kr
        xsave[3] = xsave1
        xsave[4::3] = amp
        xsave[5::3] = np.cos(arg)
        xsave[6::3] = np.sin(arg)

        # Altogether 3 + 3*(n/2)+1 elements used for q != 0, which is (3*n)/2+4
        # for even n, (3*n)/2+3 for odd n.  For even n, the very last element
        # of xsave [i.e. xsave(3*m+1)=sin(arg) for m=n/2] is not used within
        # FFTLog; if a low-ringing kr is used, this element should be zero.
        # The last element is computed in case somebody wants it.

    return kr, xsave


def fftl(a, xsave, rk=1, tdir=1):
    """Logarithmic fast Fourier transform FFTLog.

    This is a driver routine that calls fhtq.

    fftlog computes a discrete version of the Fourier sine (if mu = 1/2) or
    cosine (if mu = -1/2) transform

                        infinity
                         /
       Ã(k) = sqrt(2/pi) | A(r) sin(k r) dr
                         /
                        0

                        infinity
                         /
       Ã(k) = sqrt(2/pi) | A(r) cos(k r) dr
                         /
                        0

    by making the substitutions
                    q-(1/2)                      -q-(1/2)
       A(r) = a(r) r          and   Ã(k) = ã(k) k

    and applying a biased Hankel transform to a(r).

    The steps are:
    1. a(r) = A(r) r^[-dir*(q-.5)]
    2. call fhtq to transform a(r) -> ã(k)
    3. Ã(k) = ã(k) k^[-dir*(q+.5)]

    fhti must be called before the first call to fftlog, with mu = 1/2 for a
    sine transform, or mu = -1/2 for a cosine transform.

    A call to fftlog with dir=1 followed by a call to fftlog with dir=-1 (and
    rk unchanged), or vice versa, leaves the array a unchanged.

    Parameters
    ----------
    a : array
        Array A(r) to transform: a(j) is A(r_j) at r_j = r_c exp[(j-jc) dlnr],
        where jc = (n+1)/2 = central index of array.

    xsave : array
        Working array set up by fhti.

    rk : float, optional
        r_c/k_c = r_j/k_j (a constant, the same constant for any j); rk is not
        (necessarily) the same quantity as kr.  rk is used only to multiply the
        output array by sqrt(rk)^dir, so if you want to do the normalization
        later, or you don't care about the normalization, you can set rk = 1.
        Defaults to 1.

    tdir : int, optional; {1, -1}
        -  1 for forward transform (default),
        - -1 for backward transform.
        A backward transform (dir = -1) is the same as a forward transform with
        q -> -q and rk -> 1/rk, for any kr if n is odd, for low-ringing kr if n
        is even.

    Returns
    -------
    a : array
        Transformed array Ã(k): a(j) is Ã(k_j) at k_j = k_c exp[(j-jc) dlnr].

    """
    fct = a.copy()
    q = xsave[0]
    dlnr = xsave[1]
    kr = xsave[2]

    # centre point of array
    jc = np.array((fct.size + 1)/2.0)
    j = np.arange(fct.size)+1

    # a(r) = A(r) (r/rc)^[-dir*(q-.5)]
    fct *= np.exp(-tdir*(q - 0.5)*(j - jc)*dlnr)

    # transform a(r) -> ã(k)
    fct = fhtq(fct, xsave, tdir)

    # Ã(k) = ã(k) k^[-dir*(q+.5)] rc^[-dir*(q-.5)]
    #      = ã(k) (k/kc)^[-dir*(q+.5)] (kc rc)^(-dir*q) (rc/kc)^(dir*.5)
    lnkr = np.log(kr)
    lnrk = np.log(rk)
    fct *= np.exp(-tdir*((q + 0.5)*(j - jc)*dlnr + q*lnkr - lnrk/2.0))

    return fct


def fht(a, xsave, tdir=1):
    """Fast Hankel transform FHT.

    This is a driver routine that calls fhtq.

    fht computes a discrete version of the Hankel transform

             infinity
              /
       Ã(k) = | A(r) J  (k r) k dr
              /       mu
             0

    by making the substitutions
                    q                      -q
       A(r) = a(r) r    and   Ã(k) = ã(k) k

    and applying a biased Hankel transform to a(r).

    The steps are:
    1. a(r) = A(r) r^(-dir*q)
    2. call fhtq to transform a(r) -> ã(k)
    3. Ã(k) = ã(k) k^(-dir*q)

    fhti must be called before the first call to fht.

    A call to fht with dir=1 followed by a call to fht with dir=-1, or vice
    versa, leaves the array a unchanged.



    Parameters
    ----------
    a : array
        Array A(r) to transform: a(j) is A(r_j) at r_j = r_c exp[(j-jc) dlnr],
        where jc = (n+1)/2 = central index of array.

    xsave : array
        Working array set up by fhti.

    tdir : int, optional; {1, -1}
        -  1 for forward transform (default),
        - -1 for backward transform.
        A backward transform (dir = -1) is the same as a forward transform with
        q -> -q, for any kr if n is odd, for low-ringing kr if n is even.

    Returns
    -------
    a : array
        Transformed array Ã(k): a(j) is Ã(k_j) at k_j = k_c exp[(j-jc) dlnr].

    """
    fct = a.copy()
    q = xsave[0]
    dlnr = xsave[1]
    kr = xsave[2]

    # a(r) = A(r) (r/rc)^(-dir*q)
    if q != 0:
        #  centre point of array
        jc = np.array((fct.size + 1)/2.0)
        j = np.arange(fct.size)+1
        fct *= np.exp(-tdir*q*(j - jc)*dlnr)

    # transform a(r) -> ã(k)
    fct = fhtq(fct, xsave, tdir)

    # Ã(k) = ã(k) (k rc)^(-dir*q)
    #      = ã(k) (k/kc)^(-dir*q) (kc rc)^(-dir*q)
    if q != 0:
        lnkr = np.log(kr)
        fct *= np.exp(-tdir*q*((j - jc)*dlnr + lnkr))

    return fct


def fhtq(a, xsave, tdir=1):
    """Kernel routine of FFTLog.

    This is the basic FFTLog routine.

    fhtq computes a discrete version of the biased Hankel transform

             infinity
              /           q
       ã(k) = | a(r) (k r)  J  (k r) k dr
              /              mu
             0

    fhti must be called before the first call to fhtq.

    A call to fhtq with dir=1 followed by a call to fhtq with dir=-1, or vice
    versa, leaves the array a unchanged.

    Parameters
    ----------
    a : array
        Periodic array a(r) to transform: a(j) is a(r_j) at r_j = r_c
        exp[(j-jc) dlnr] where jc = (n+1)/2 = central index of array.

    xsave : array
        Working array set up by fhti.

    tdir : int, optional; {1, -1}
        -  1 for forward transform (default),
        - -1 for backward transform.
        A backward transform (dir = -1) is the same as a forward transform with
        q -> -q, for any kr if n is odd, for low-ringing kr if n is even.

    Returns
    -------
    a : array
        Transformed periodic array ã(k): a(j) is ã(k_j) at k_j = k_c exp[(j-jc)
        dlnr].

    """
    fct = a.copy()
    q = xsave[0]
    n = fct.size

    # normal FFT
    fct = rfft(fct)

    m = np.arange(1, n//2, dtype=int)  # index variable
    if q == 0:  # unbiased (q = 0) transform
        # multiply by (kr)^[- i 2 m pi/(n dlnr)] U_mu[i 2 m pi/(n dlnr)]
        ar = fct[2*m-1]
        ai = fct[2*m]
        fct[2*m-1] = ar*xsave[2*m+1] - ai*xsave[2*m+2]
        fct[2*m] = ar*xsave[2*m+2] + ai*xsave[2*m+1]
        # problem(2*m)atical last element, for even n
        if np.mod(n, 2) == 0:
            ar = xsave[-2]
            if (tdir == 1):  # forward transform: multiply by real part
                # Why? See http://casa.colorado.edu/~ajsh/FFTLog/index.html#ure
                fct[-1] *= ar
            elif (tdir == -1):  # backward transform: divide by real part
                # Real part ar can be zero for maximally bad choice of kr.
                # This is unlikely to happen by chance, but if it does, policy
                # is to let it happen.  For low-ringing kr, imaginary part ai
                # is zero by construction, and real part ar is guaranteed
                # nonzero.
                fct[-1] /= ar

    else:  # biased (q != 0) transform
        # multiply by (kr)^[- i 2 m pi/(n dlnr)] U_mu[q + i 2 m pi/(n dlnr)]
        # phase
        ar = fct[2*m-1]
        ai = fct[2*m]
        fct[2*m-1] = ar*xsave[3*m+2] - ai*xsave[3*m+3]
        fct[2*m] = ar*xsave[3*m+3] + ai*xsave[3*m+2]

        if tdir == 1:  # forward transform: multiply by amplitude
            fct[0] *= xsave[3]
            fct[2*m-1] *= xsave[3*m+1]
            fct[2*m] *= xsave[3*m+1]

        elif tdir == -1:  # backward transform: divide by amplitude
            # amplitude of m=0 element
            ar = xsave[3]
            if ar == 0:
                # Amplitude of m=0 element can be zero for some mu, q
                # combinations (singular inverse); policy is to drop
                # potentially infinite constant.
                fct[0] = 0
            else:
                fct[0] /= ar

            # remaining amplitudes should never be zero
            fct[2*m-1] /= xsave[3*m+1]
            fct[2*m] /= xsave[3*m+1]

        # problematical last element, for even n
        if np.mod(n, 2) == 0:
            m = int(n/2)
            ar = xsave[3*m+2]*xsave[3*m+1]
            if tdir == 1:  # forward transform: multiply by real part
                fct[-1] *= ar
            elif (tdir == -1):  # backward transform: divide by real part
                # Real part ar can be zero for maximally bad choice of kr.
                # This is unlikely to happen by chance, but if it does, policy
                # is to let it happen.  For low-ringing kr, imaginary part ai
                # is zero by construction, and real part ar is guaranteed
                # nonzero.
                fct[-1] /= ar

    # normal FFT back
    fct = irfft(fct)

    # reverse the array and at the same time undo the FFTs' multiplication by n
    # => Just reverse the array, the rest is already done in drfft.
    fct = fct[::-1]

    return fct


def krgood(mu, q, dlnr, kr):
    """Return optimal kr

    Use of this routine is optional.

    Choosing kr so that
        (kr)^(- i pi/dlnr) U_mu(q + i pi/dlnr)
    is real may reduce ringing of the discrete Hankel transform, because it
    makes the transition of this function across the period boundary smoother.

    Parameters
    ----------
    mu : float
        index of J_mu in Hankel transform; mu may be any real number, positive
        or negative.

    q : float
        exponent of power law bias; q may be any real number, positive or
        negative.  If in doubt, use q = 0, for which case the Hankel transform
        is orthogonal, i.e. self-inverse, provided also that, for n even, kr is
        low-ringing.  Non-zero q may yield better approximations to the
        continuous Hankel transform for some functions.

    dlnr : float
        separation between natural log of points; dlnr may be positive or
        negative.

    kr : float, optional
        k_c r_c where c is central point of array
        = k_j r_(n+1-j) = k_(n+1-j) r_j .
        Normally one would choose kr to be about 1 (default) (or 2, or pi, to
        taste).

    Returns
    -------
    krgood : float
        low-ringing value of kr nearest to input kr.  ln(krgood) is always
        within dlnr/2 of ln(kr).

    """
    if dlnr == 0:
        return kr

    xp = (mu + 1.0 + q)/2.0
    xm = (mu + 1.0 - q)/2.0
    y = 1j*np.pi/(2.0*dlnr)
    zp = loggamma(xp + y)
    zm = loggamma(xm + y)

    # low-ringing condition is that following should be integral
    arg = np.log(2.0/kr)/dlnr + (zp.imag + zm.imag)/np.pi

    # return low-ringing kr
    return kr*np.exp((arg - np.round(arg))*dlnr)
