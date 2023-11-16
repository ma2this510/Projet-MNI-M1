import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N = 32

df = pd.read_csv('energies_alpha.dat', sep='\t', header=None)
df.columns = ['t', 'energy', 'ek0', 'ek1', 'ek2', 'ek3']

df['x'] = np.sin(np.pi/(2*N)) * df['t'] / np.pi

plt.plot(df['x'], df['energy'], label='Total energy')
plt.plot(df['x'], df['ek0'], label='Kinetic energy 1')
plt.plot(df['x'], df['ek1'], label='Kinetic energy 2')
plt.plot(df['x'], df['ek2'], label='Kinetic energy 3')
plt.plot(df['x'], df['ek3'], label='Kinetic energy 4')
# plt.legend()
plt.text(1450, 0.07, r'$1$', color='C1', fontsize=14)
plt.text(715, 0.06, r'$2$', color='C2', fontsize=14)
plt.text(970, 0.04, r'$3$', color='C3', fontsize=14)
plt.text(315, 0.032, r'$4$', color='C4', fontsize=14)
plt.xlabel(r'$w_1 t /2\pi$')
plt.ylabel(r'$E_k \times 10^2$')
plt.title("Evolution of the energies for $\\alpha = 0.25$")
plt.tight_layout()
plt.savefig('fput_0.25.pdf')
plt.show()
