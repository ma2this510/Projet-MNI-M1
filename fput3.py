import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Nombre d'oscillateurs
N = 32

# Chargement des données
df = pd.read_csv('energies_alpha.dat', sep='\t', header=None)
df.columns = ['t', 'energy', 'ek0', 'ek1', 'ek2', 'ek3']

# Conversion des énergies en unités de 10^-2
df['energy'] = df['energy'] * 100
df['ek0'] = df['ek0'] * 100
df['ek1'] = df['ek1'] * 100
df['ek2'] = df['ek2'] * 100
df['ek3'] = df['ek3'] * 100

# Conversion du temps en unités de w_1 t / 2pi
df['x'] = np.sin(np.pi/(2*N)) * df['t'] / (np.pi)

# Tracé des graphiques
plt.plot(df['x'], df['energy'], label='Total energy')
plt.plot(df['x'], df['ek0'], label='Kinetic energy 1')
plt.plot(df['x'], df['ek1'], label='Kinetic energy 2')
plt.plot(df['x'], df['ek2'], label='Kinetic energy 3')
plt.plot(df['x'], df['ek3'], label='Kinetic energy 4')

# Légende
plt.text(145, 7, r'$1$', color='C1', fontsize=14)
plt.text(71.5, 6, r'$2$', color='C2', fontsize=14)
plt.text(97, 4, r'$3$', color='C3', fontsize=14)
plt.text(31.5, 3.2, r'$4$', color='C4', fontsize=14)

# Paramètres du graphique
plt.xlabel(r'$w_1 t /2\pi$')
plt.ylabel(r'$E_k \times 10^2$')
plt.title("Evolution of the energies for $\\alpha = 0.25$")
plt.tight_layout()
plt.savefig('fput_0.25.pdf')
plt.show()
