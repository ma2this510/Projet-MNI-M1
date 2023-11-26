import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


def r(x):
    return x*(np.sqrt(2*np.pi)/4.0)*np.exp(-x**2/4.0)*(sp.iv(0, x**2/4.0) + sp.iv(1, x**2/4.0))


g_prime_prime = -1/np.sqrt(2*np.pi)
g_prime = 0
g = 1/np.sqrt(2*np.pi)
k_c = 2/(np.pi * g)


def r_approx(x): return np.sqrt(-16*(x - k_c)/(np.pi*(k_c**4)*g_prime_prime))


y = np.linspace(0.01, 1.2, 120)
k = y/r(y)

plt.plot(k, r(y), label='exact')
plt.plot(k, r_approx(k), label='approx')
plt.xlabel('k')
plt.ylabel('r(y)')
plt.legend()
plt.tight_layout()
plt.savefig('traceexact2.pdf')
plt.show()
