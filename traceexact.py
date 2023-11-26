import matplotlib.pyplot as plt
import numpy as np

g_prime_prime = -1/np.sqrt(2*np.pi)
g_prime = 0
g = 1/np.sqrt(2*np.pi)
k_c = 2/(np.pi * g)


def r(x): return np.sqrt(-16*(x - k_c)/(np.pi*(k_c**4)*g_prime_prime))


x = np.linspace(0, 1.8, 1000)
y = r(x)

print(r(1.8))
plt.plot(x, y)
plt.xlabel(r'$K$')
plt.ylabel(r"$r = \sqrt{\frac{-16 (K - K_c)}{\pi K_c^4 g''(0)}}$")
plt.title(r"$r$ en fonction de $K$")
plt.tight_layout()
plt.savefig('traceexacte.pdf')
plt.show()
