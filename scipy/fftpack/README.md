Status of incorporation of `FFTLog` into `fftpack`
==================================================


Done
----
- `basic.py`
  - `atexit`: destroy fftlog cache
  - `_raw_fftlog` aux-fct for `rfftl`, `irfftl`, `rfht`, `irfht`
  - `rfftl`: Logarithmic fast Fourier transform; forward
  - `rfht` : Logarithmic fast Hankel transform; forward

- `fftpack.pyf`
  - `subroutine drfht`
  - `subroutine destroy_fftlog_cache`

- `src/drfft.c`
  - `F_FUNC` for `fftl`, `fht`, and `fhti`.
  - `GEN_CACHE(fftlog,...)`
  - `drfftl`
  - `drfht`

- `setup.py`: `fftlog_src` and `fftlog`-library

- `__init__.py`: Added `rfftl`, `rfht`

- `src/fftlog/fftlog.f`: Adjusted for inclusion, all changes are marked with
  `%DW`
  - `drfft*` -> `dfft*`
  - `kr`: remove verbose and interactive options (`kropt={0,1}`)
    - remove parameters : `ok`, `lnblnk`, `stblnk`, `go`, `temp`
    - remove function   : `stblnk(string)`
  - `rk` in `fftl`: replace `lnrk=log(rk)` with `lnrk=log(rk/kr)`; this way,
    one does not have to know the new value of `kr` from `wsave`.
- Simple notebooks to test the implementation, in `scipy/scipy/fftpack/.`


ToDo's
------
- `basic.py`
  - `_raw_fftlog`, `rfftl`, `rfht` : proper implementation and documentation
  - implement `irfftl` and `irfht` (inverse fct's of  `rfftl` and `rfht`)
- `__init__.py`: Documentation
- Replace `cdgamma` with `scipy.special.loggamma` (asked Joshua W. for help)
- Add tests
- Adjust: `bento.info`, `bscript`, `fftpack_version.py`, `MANIFEST.in`,
  `NOTES.txt`
- The new `kr`-values has to be returned, if adjusted! (`krgood`)


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

