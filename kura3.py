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
    __init__(self, N, K)
        Initialise la classe OSCI avec les paramètres donnés.
    KURA(self, t, omega)
        Calcule la dérivée des angles de phase des oscillateurs.
    solve(self, tmax, step)
        Résout l'équation différentielle pour la plage de temps donnée.
    graph(self)
        Trace l'état actuel des oscillateurs.
    """

    def __init__(self, N, K):
        """
        Initialise la classe OSCI avec les paramètres donnés.

        Paramètres:
        N (int): Le nombre d'oscillateurs.
        K (float): La force de couplage.
        """
        self.N = N
        self.K = K
        self.t_n = 0

        self.pulse = np.random.normal(0, 1, N)

        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega):
        """
        Calcule la dérivée des angles de phase des oscillateurs.

        Paramètres:
        t (float): Le temps actuel.
        omega (ndarray): Les angles de phase actuels des oscillateurs.

        Retourne:
        ndarray: La dérivée des angles de phase des oscillateurs.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        self.abs_list = np.append(self.abs_list, np.abs(ordre))
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Résout l'équation différentielle pour la plage de temps donnée.

        Paramètres:
        tmax (float): Le temps maximum à résoudre.
        step (int): Le nombre d'étapes de temps à utiliser.

        Retourne:
        OdeResult: La solution de l'équation différentielle.
        """
        t = np.linspace(self.t_n, tmax, step)

        self.abs_list = np.empty(1)
        self.abs_list[0] = np.sum(np.abs(self.pulse))
        sol = solve_ivp(fun=self.KURA, t_span=(
            self.t_n, tmax), y0=self.omega, t_eval=t)
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += t[-1]

        return sol

    def graph(self):
        """
        Trace l'état actuel des oscillateurs.

        Crée un graphique de dispersion de l'état actuel des oscillateurs, avec l'axe x représentant le cosinus de la phase de l'oscillateur et l'axe y représentant le sinus de la phase de l'oscillateur. Le graphique inclut également un cercle de rayon 1 centré à l'origine, qui représente le cercle unité. Le paramètre d'ordre actuel est tracé en tant que point vert, et les valeurs actuelles de K et t_n sont incluses dans le titre du graphique. Le graphique résultant est enregistré sous forme de fichier PDF et affiché.
        """
        circle = plt.Circle((0, 0), 1, fill=False, color='r')
        fig, axs = plt.subplots(1, 1)
        axs.add_artist(circle)
        axs.set_xlim(-1.5, 1.5)
        axs.set_ylim(-1.5, 1.5)
        axs.scatter(np.cos(self.omega), np.sin(self.omega), color='b')
        axs.scatter(np.real(self.ordre), np.imag(self.ordre), color='g')
        axs.set_aspect('equal')
        axs.grid(True, which='both')
        axs.set_title(f'K = {self.K} et t = {self.t_n}')
        axs.set_xlabel(r'$\cos$ et $real$')
        axs.set_ylabel(r'$\sin$ et $imag$')
        # axs[1].plot(np.linspace(0, self.t_n, len(
        #     self.abs_list))[1:], self.abs_list[1:])
        # axs[1].set_xlabel('t')
        # axs[1].set_ylabel(r'$r(t)$')
        # axs[1].set_title(r'Evolutions de $r$ en fonction de $t$')
        plt.tight_layout()
        plt.savefig(f'kura3_{self.N}.pdf')
        plt.show()


oscis = OSCI(100, 2)
sol = oscis.solve(100, 201)
oscis.graph()
