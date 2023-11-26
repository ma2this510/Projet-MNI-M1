import matplotlib.pyplot as plt
import numpy as np

np.random.seed(40)


class OSCI:
    def __init__(self, N):
        """
        Classe représentant un oscillateur.

        Args:
            N (int): Le nombre d'oscillateurs.

        Attributes:
            N (int): Le nombre d'oscillateurs.
            omega (ndarray): Les valeurs aléatoires des phases des oscillateurs.
            ordre (complex): L'ordre moyen des oscillateurs.
        """
        self.N = N
        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def graph(self):
        """
        Affiche le graphique des oscillateurs.

        Returns:
            None
        """
        circle = plt.Circle((0, 0), 1, fill=False, color='r')
        fig, ax = plt.subplots()
        ax.add_artist(circle)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.scatter(np.cos(self.omega), np.sin(self.omega), color='b')
        ax.scatter(np.real(self.ordre), np.imag(self.ordre), color='g')
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        plt.savefig(f'kura1_{self.N}.pdf')
        plt.show()


oscis = OSCI(10)
oscis.graph()
