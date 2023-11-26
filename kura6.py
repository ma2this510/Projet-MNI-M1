import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(40)


class OSCI:
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
        self.pulse -= np.mean(self.pulse)

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

        sol = solve_ivp(fun=self.KURA, t_span=(
            self.t_n, tmax), y0=self.omega, t_eval=t)

        self.sol = sol
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += t[-1]

        return sol

    def graph(self):
        """
        Trace l'état actuel des oscillateurs.

        Crée un graphique de dispersion de l'état actuel des oscillateurs, l'axe des x représentant le cosinus de la phase de l'oscillateur et l'axe des y représentant le sinus de la phase de l'oscillateur. Le graphique inclut également un cercle de rayon 1 centré à l'origine, qui représente le cercle unité. Le paramètre d'ordre actuel est tracé en tant que point vert, et les valeurs actuelles de K et t_n sont incluses dans le titre du graphique. Le graphique résultant est enregistré en tant que fichier PDF et affiché.
        """
        circle = plt.Circle((0, 0), 1, fill=False, color='r')
        fig, axs = plt.subplots(1, 2)
        axs[0].add_artist(circle)
        axs[0].set_xlim(-1.5, 1.5)
        axs[0].set_ylim(-1.5, 1.5)
        axs[0].scatter(np.cos(self.omega), np.sin(self.omega), color='b')
        axs[0].scatter(np.real(self.ordre), np.imag(self.ordre), color='g')
        axs[0].set_aspect('equal')
        axs[0].grid(True, which='both')
        axs[0].set_title(f'K = {self.K} et t = {self.t_n}')
        axs[0].set_xlabel(r'$\cos$ et $real$')
        axs[0].set_ylabel(r'$\sin$ et $imag$')
        axs[1].plot(np.linspace(0, self.t_n, len(
            self.abs_list)), self.abs_list)
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('abs(ordre)')
        axs[1].set_title('$r(t)$')
        axs[1].set_aspect('equal')
        plt.tight_layout()
        # plt.savefig(f'kura4_{self.N}.pdf')
        plt.show()

    def get_abs_ordre(self):
        self.abs_list = np.abs(
            np.sum(np.exp(1j * self.sol.y[:, sol.t >= 50]), axis=0)) / self.N
        return self.abs_list


# Initialisation des paramètres
k_list = np.linspace(1, 2, 51)
Nrep = 50
abs_tot = np.empty((len(k_list), Nrep))

# Calcul de abs(ordre) pour chaque valeur de K
for i, k in enumerate(k_list):
    for j in range(Nrep):
        oscis = OSCI(100, k)
        sol = oscis.solve(100, 201)
        abs_tot[i, j] = np.mean(oscis.get_abs_ordre())

mean_abs_tot = np.mean(abs_tot, axis=1)

# Tracé de abs(ordre) en fonction de K
plt.plot(k_list, mean_abs_tot)
plt.xlabel(r'$K$')
plt.ylabel(r'$| r |$')
plt.title(f'Moyenne de $| r |$ en fonction de $K$ : {Nrep} répétitions')
plt.tight_layout()
plt.savefig(f'kura6_{Nrep}.pdf')
plt.show()
