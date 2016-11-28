from scipy.fftpack import fftlog
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mpl.rcParams.update({'font.size': 16})


## Define the parameters you wish to use
# The presets are the *Reasonable choices of parameters* from `fftlogtest.f`.

# Range of periodic interval
logrmin = -4
logrmax = 4

# Number of points (Max 4096)
n = 64

# Order mu of Bessel function
mu = 0

# Bias exponent: q = 0 is unbiased
q = 0

# Sensible approximate choice of k_c r_c
kr = 1

# Tell fhti to change kr to low-ringing value
# WARNING: kropt = 3 will fail, as interaction is not supported
kropt = 1

# Forward transform (changed from dir to tdir, as dir is a python fct)
tdir = 1

## Calculation related to the logarithmic spacing
# Central point log10(r_c) of periodic interval
logrc = (logrmin + logrmax)/2

print('Central point of periodic interval at log10(r_c) = ', logrc)

# Central index (1/2 integral if n is even)
nc = (n + 1)/2.0

# Log-spacing of points
dlogr = (logrmax - logrmin)/n
dlnr = dlogr*np.log(10.0)


## Calculate input function: $r^{\mu+1}\exp\left(-\frac{r^2}{2}\right)$
r = 10**(logrc + (np.arange(1, n+1) - nc)*dlogr)
ar = r**(mu + 1)*np.exp(-r**2/2.0)


## Initialize FFTLog transform - note fhti resets `kr`
kr, wsave, ok = fftlog.fhti(n, mu, dlnr, kr, q, kropt)
print('fftlog.fhti: ok =', bool(ok), '; New kr = ', kr)


## Call `fftlog.fht` (or `fftlog.fhtl`)
logkc = np.log10(kr) - logrc
print('Central point in k-space at log10(k_c) = ', logkc)

# rk = r_c/k_c
rk = 10**(logrc - logkc)

# Transform
#ak = fftlog.fftl(ar, tdir, wsave, rk)
ak = fftlog.fht(ar.copy(), tdir, wsave)

## Calculate Output function: $k^{\mu+1}\exp\left(-\frac{k^2}{2}\right)$
k = 10**(logkc + (np.arange(1, n+1) - nc)*dlogr)
theo = k**(mu + 1)*np.exp(-k**2/2.0)


## Plot result
plt.figure(figsize=(16,8))

# Transformed result
ax2 = plt.subplot(1, 2, 2)
plt.plot(k, theo, 'k', lw=2, label='Theoretical')
plt.plot(k, ak, 'r--', lw=2, label='FFTLog')
plt.xlabel('k')
plt.title(r'$k^{\mu+1} \exp(-k^2/2)$', fontsize=20)
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('symlog', basey=10, linthreshy=1e-5)
ax2ylim = plt.ylim()

# Input
ax1 = plt.subplot(1, 2, 1)
plt.plot(r, ar, 'k', lw=2)
plt.xlabel('r')
plt.title(r'$r^{\mu+1}\ \exp(-r^2/2)$', fontsize=20)
plt.xscale('log')
plt.yscale('symlog', basey=10, linthreshy=1e-5)
plt.ylim(ax2ylim)

# Main title
plt.suptitle(r'$\int_0^\infty r^{\mu+1}\ \exp(-r^2/2)\ J_\mu(k,r)\ k\ {\rm d}r = k^{\mu+1} \exp(-k^2/2)$',
             fontsize=24, y=1.08)
plt.show()


## Print values
print('           k                 a(k)       k^(mu+1) exp(-k^2/2)')
print('----------------------------------------------------------------')
for i in range(n):
    print("%18.6e %18.6e %18.6e"% (k[i], ak[i], theo[i]))
