/*
  Interface to various FFT libraries.
  Double real FFT and IFFT.
  Author: Pearu Peterson, August 2002
 */

#include "fftpack.h"

extern void F_FUNC(dfftf, DFFTF) (int *, double *, double *);
extern void F_FUNC(dfftb, DFFTB) (int *, double *, double *);
extern void F_FUNC(dffti, DFFTI) (int *, double *);
extern void F_FUNC(rfftf, RFFTF) (int *, float *, float *);
extern void F_FUNC(rfftb, RFFTB) (int *, float *, float *);
extern void F_FUNC(rffti, RFFTI) (int *, float *);
extern void F_FUNC(fftl, FFTL) (int *, double *, double *, int *, double *,
                                double *);
extern void F_FUNC(fhti, FHTI) (int *, double *, double *, double *, double *,
                                double *);

GEN_CACHE(drfft, (int n)
      , double *wsave;
      , (caches_drfft[i].n == n)
      , caches_drfft[id].wsave =
      (double *) malloc(sizeof(double) * (2 * n + 15));
      F_FUNC(dffti, DFFTI) (&n, caches_drfft[id].wsave);
      , free(caches_drfft[id].wsave);
      , 10)

GEN_CACHE(rfft, (int n)
      , float *wsave;
      , (caches_rfft[i].n == n)
      , caches_rfft[id].wsave =
      (float *) malloc(sizeof(float) * (2 * n + 15));
      F_FUNC(rffti, RFFTI) (&n, caches_rfft[id].wsave);
      , free(caches_rfft[id].wsave);
      , 10)

GEN_CACHE(fftlog
      , (int n, double mu, double q, double dlnr, double kr, int size)
      , double *xsave; double mu; double q; double dlnr; double kr;
      , ((caches_fftlog[i].n == n) && (caches_fftlog[i].mu == mu) &&
         (caches_fftlog[i].q == q) && (caches_fftlog[i].dlnr == dlnr) &&
         (caches_fftlog[i].kr == kr))
      , caches_fftlog[id].xsave = (double *)
        malloc(sizeof(double) * size);
        F_FUNC(fhti, FHTI) (&n, &mu, &q, &dlnr, &kr, caches_fftlog[id].xsave);
      , free(caches_fftlog[id].xsave);
      , 10)


void drfft(double *inout, int n, int direction, int howmany,
              int normalize)
{
    int i;
    double *ptr = inout;
    double *wsave = NULL;
    wsave = caches_drfft[get_cache_id_drfft(n)].wsave;


    switch (direction) {
        case 1:
        for (i = 0; i < howmany; ++i, ptr += n) {
            F_FUNC(dfftf,DFFTF)(&n, ptr, wsave);
        }
        break;

    case -1:
        for (i = 0; i < howmany; ++i, ptr += n) {
            F_FUNC(dfftb,DFFTB)(&n, ptr, wsave);
        }
        break;

    default:
        fprintf(stderr, "drfft: invalid direction=%d\n", direction);
    }

    if (normalize) {
        double d = 1.0 / n;
        ptr = inout;
        for (i = n * howmany - 1; i >= 0; --i) {
            (*(ptr++)) *= d;
        }
    }
}

void rfft(float *inout, int n, int direction, int howmany,
             int normalize)
{
    int i;
    float *ptr = inout;
    float *wsave = NULL;
    wsave = caches_rfft[get_cache_id_rfft(n)].wsave;


    switch (direction) {
        case 1:
        for (i = 0; i < howmany; ++i, ptr += n) {
            F_FUNC(rfftf,RFFTF)(&n, ptr, wsave);
        }
        break;

    case -1:
        for (i = 0; i < howmany; ++i, ptr += n) {
            F_FUNC(rfftb,RFFTB)(&n, ptr, wsave);
        }
        break;

    default:
        fprintf(stderr, "rfft: invalid direction=%d\n", direction);
    }

    if (normalize) {
        float d = 1.0 / n;
        ptr = inout;
        for (i = n * howmany - 1; i >= 0; --i) {
            (*(ptr++)) *= d;
        }
    }
}

void drfftl(double *inout, int n, double mu, double q, double dlnr, double kr,
            double rk, int direction, int howmany)
{
    int i;
    int size;
    double *ptr = inout;
    double *xsave = NULL;
    double *wsave = NULL;

    if ( q != 0) {
        size = 3 * (n / 2) + 4;
    }
    else {
        size = 2 * (n / 2) + 3;
    }

    wsave = caches_drfft[get_cache_id_drfft(n)].wsave;
    xsave = caches_fftlog[get_cache_id_fftlog(n, mu, q, dlnr, kr, size)].xsave;

    for (i = 0; i < howmany; ++i, ptr += n) {
        F_FUNC(fftl, FFTL)(&n, ptr, &rk, &direction, wsave, xsave);
    }
}
