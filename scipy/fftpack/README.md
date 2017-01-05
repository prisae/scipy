Todo's and questions regarding the incorporation of `FFTLog` into `fftpack`
===========================================================================

Examples of the usage and to test the implementation are given in the notebook
`scipy/scipy/fftpack/FFTLog-Examples.ipynb`.


ToDo's
------
- Implement the reverse function `irfftl`.
- Implement type-check.
- Implement complex version.
- `__init__.py`: Documentation
- Replace `cdgamma` with `scipy.special.loggamma` (asked Joshua W. for help)
- Add tests
- Adjust: `bento.info`, `bscript`, `fftpack_version.py`, `MANIFEST.in`,
  `NOTES.txt`


Open questions
--------------
- `fftl` is the original name; it might be better to call it `dctlog` and
  `dstlog`, in analogy to `dct` and `dst`?
- Are the implementations of `overwrite_x` (`n`, `axis`) correct? (Copied from
  `rfft`, but does it apply to `rfftl`?
- Pearu P. suggests to move `drfftl` from `src/drfft.c` into own `src/fftlog.c`
  file; however, it shares with the traditional `drfft` `caches_drfft`, and as
  such `dffti` and `wsave` dffti. So I am not exactly sure how to do that.


New file
--------
- `src/fftlog/fftlog.f`


Changed files
-------------
- `__init__.py`
- `basic.py`
- `fftpack.pyf`
- `helper.py`
- `setup.py`
- `src/drfft.c`


Temporary files
---------------
- `src/fftlag/cdgamma.f` (will be replaced with `scipy.special.loggamma`)
- `FFTLog-Examples.ipynb`
- `README.md` (this file)


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

