Status of incorporation of `FFTLog` into `fftpack`
==================================================


Done
----
- `basic.py`
  - `atexit`: destroy fftlog cache
  - `_raw_fftlog` aux-fct for `rfftl`
  - `rfftl`: Logarithmic fast Fourier transform; forward

- `fftpack.pyf`
  - `subroutine drfht`
  - `subroutine getkr`
  - `subroutine destroy_fftlog_cache`

- `src/drfft.c`
  - `F_FUNC` for `fftl` and `fhti`.
  - `GEN_CACHE(fftlog,...)`
  - `drfftl`

- `setup.py`: `fftlog_src` and `fftlog`-library

- `__init__.py`: Added `rfftl`

- `src/fftlog/fftlog.f`: Adjusted for inclusion, all changes are marked with
  `%DW`
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

- Simple notebook `scipy/scipy/fftpack/FFTLog-Examples.ipynb` to test the
  implementation.


ToDo's
------
- `basic.py`
  - `_raw_fftlog`, `rfftl` : proper implementation and documentation
  - implement `irfftl` (and `rfht`/`irfht`?)
- `__init__.py`: Documentation
- Replace `cdgamma` with `scipy.special.loggamma` (asked Joshua W. for help)
- Add tests
- Adjust: `bento.info`, `bscript`, `fftpack_version.py`, `MANIFEST.in`,
  `NOTES.txt`


Questions
---------
- id's in `src/drfft.c` => `GEN_CACHE`?


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

