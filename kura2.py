import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(40)


class OSCI:
    """
    Une classe représentant un système d'oscillateurs.

    Attributs
    ----------
    N : int
        Le nombre d'oscillateurs dans le système.
    K : float
        La force de couplage entre les oscillateurs.
    t_n : float
        Le temps actuel du système.
    pulse : numpy.ndarray
        Un tableau de valeurs aléatoires représentant l'impulsion initiale de chaque oscillateur.
    omega : numpy.ndarray
        Un tableau de valeurs aléatoires représentant la fréquence initiale de chaque oscillateur.
    ordre : complex
        Le paramètre d'ordre du système, calculé comme la moyenne de l'exponentielle complexe des fréquences des oscillateurs.

    Méthodes
    -------
    KURA(t, omega)
        Calcule la dérivée des fréquences des oscillateurs à un temps donné t et une fréquence omega.
    solve(tmax, step)
        Résout le système en utilisant la fonction KURA et la méthode Runge-Kutta 45.
    graph()
        Trace l'état actuel du système sur un plan complexe et enregistre le graphique au format PDF.
    """

    def __init__(self, N, K):
        """
        Initialise la classe OSCI.

        Paramètres
        ----------
        N : int
            Le nombre d'oscillateurs dans le système.
        K : float
            La force de couplage entre les oscillateurs.
        """
        self.N = N
        self.K = K
        self.t_n = 0
        self.pulse = np.random.normal(0, 1, N)
        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega):
        """
        Calcule la dérivée des fréquences des oscillateurs à un temps donné t et une fréquence omega.

        Paramètres
        ----------
        t : float
            Le temps actuel du système.
        omega : numpy.ndarray
            Un tableau de valeurs représentant la fréquence actuelle de chaque oscillateur.

        Retour
        -------
        numpy.ndarray
            Un tableau de valeurs représentant la dérivée de la fréquence de chaque oscillateur.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Résout le système en utilisant la fonction KURA et la méthode Runge-Kutta 45.

        Paramètres
        ----------
        tmax : float
            Le temps maximum pour résoudre le système.
        step : int
            Le nombre d'étapes à utiliser dans le solveur.

        Retour
        -------
        scipy.integrate.OdeResult
            Un objet contenant la solution du système.
        """
        t = np.linspace(self.t_n, tmax, step)
        sol = solve_ivp(fun=self.KURA, t_span=(
            self.t_n, tmax), y0=self.omega, t_eval=t)
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += tmax

        return sol  # Sert à rien mais bon

    def graph(self):
        """
        Trace l'état actuel du système sur un plan complexe et enregistre le graphique au format PDF.
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
        ax.set_title(f'K = {self.K} et t = {self.t_n}')
        plt.savefig(f'kura2_{self.N}.pdf')
        plt.show()


oscis = OSCI(100, 0)
sol = oscis.solve(100, 201)
oscis.graph()
