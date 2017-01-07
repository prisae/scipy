Todo's and questions regarding the incorporation of `FFTLog` into `fftpack`
===========================================================================

Examples of the usage and to test the implementation are given in the notebook
`scipy/scipy/fftpack/FFTLog-Examples.ipynb`.


ToDo's
------
- Add tests for `fftlog`
- Add CSEM example
- Replace `cdgamma` with `scipy.special.loggamma` (asked Joshua W. for help)


Notes
-----
- `fftlog` has not, as other implemented FFT routines, any of the following
  implementations:
    - type-specific versions (output is always a float64/complex128)
    - axis-check and similar (arguments `axis`, `n`, `overwrite_x`, `howmany`
      for instance in `fft` or `rfft`.)
- Cache `fftl_w` in `src/fftlog.c` is exactly the same as cache `drfft` in
  `src/drfft.c`, and could potentially be combined.
- I never used the inverse fftlog, and never got it to work.


Changes to `fftlog.f`
---------------------
Adjusted for inclusion, all changes are marked with `%DW`
- `drfft*` -> `dfft*`
- `kr`: remove `kropt` from `fhti` (see `getkr`)
  - `kr` has to be defined exactly before calling `fhti`
  - remove parameters : `kropt`, `ok`, `lnblnk`, `stblnk`, `go`, `temp`
  - remove function   : `stblnk(string)`
- `krgood`: change from `function krgood` to `subroutine getkr`
- split `wsave` into `wsave` and `xsave`
  - `wsave` is as in regular FFT
  - `xsave` is the fftlog-addition to `wsave`
  - moved `dffti` from `fhti` to `fhtq`


Dummy code
----------

    function fhti:                                # Initializing routine:
        calc.                                     #
        call irfft  <= irfft from dfftpack        # Appends more stuff to
        calc.                                     # wsave from irfft

    function fftl:                                # Fast Fourier Transform
        calc.
        call fhtq -|
        calc.      |
                   |
    function fht:  |                              # Fast Hankel Transform
        calc.      |
        call fhtq -|
        calc.      |
               /---/
               |
               v
    function fhtq:                                # Core routine
        call rfftf  <= rfftf from dfftpack
        calc.
        call rfftb  <= rfftb from dfftpack

