import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(40)


class OSCI:
    """
    Représente un système d'oscillateurs.

    Attributs:
    N (int): Le nombre d'oscillateurs.
    K (float): La force de couplage.
    t_n (float): Le temps actuel.
    pulse (ndarray): Les valeurs d'impulsion pour chaque oscillateur.
    omega (ndarray): Les angles de phase des oscillateurs.
    ordre (complex): Le paramètre d'ordre du système.
    abs_list (ndarray): La liste des valeurs absolues du paramètre d'ordre à chaque pas de temps.
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
        self.pulse -= np.mean(self.pulse)

        # Attention aux sens physique : dernier termes non tiré selon la distribution
        self.pulse[-1] = - np.sum(self.pulse[:-1])

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
        step (int): Le nombre de pas de temps à utiliser.

        Retourne:
        OdeResult: La solution de l'équation différentielle.
        """
        t = np.linspace(self.t_n, tmax, step)

        self.abs_list = np.empty(0)
        sol = solve_ivp(fun=self.KURA, t_span=(
            self.t_n, tmax), y0=self.omega, t_eval=t)
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += t[-1]

        return sol

    def get_abs_ordre(self):
        """
        Retourne la liste des valeurs absolues du paramètre d'ordre à chaque pas de temps.

        Retourne:
        ndarray: La liste des valeurs absolues du paramètre d'ordre.
        """
        return self.abs_list


for k in np.linspace(1, 2, 6):
    oscis = OSCI(100, k)
    sol = oscis.solve(100, 201)
    tmp = oscis.get_abs_ordre()
    plt.plot(np.linspace(0, 100, len(tmp)), tmp, label=f'{k}')

plt.xlabel(r'$t$')
plt.ylabel(r'$r$')
plt.legend(title=r'$K =$', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(r'Évolution de $r$ en fonction de $t$')
plt.tight_layout()
plt.savefig('kura4.pdf')
plt.show()
