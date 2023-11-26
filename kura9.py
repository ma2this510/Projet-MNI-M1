import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.special as sp
from joblib import Parallel, delayed
from tqdm import tqdm
import time

np.random.seed(40)


class OSCI:
    def __init__(self, N, K):
        """
        Initialise la classe OSCI.

        Paramètres :
        - N (int) : Nombre d'oscillateurs.
        - K (float) : Force de couplage.

        Attributs :
        - N (int) : Nombre d'oscillateurs.
        - K (float) : Force de couplage.
        - pulse (ndarray) : Tableau d'impulsions aléatoires.
        - omega (ndarray) : tableau de fréquences angulaires aléatoires : Tableau de fréquences angulaires aléatoires.
        - ordre (complexe) : Paramètre d'ordre.
        - sol (scipy.integrate.OdeSolution) : Solution de l'équation différentielle.
        """
        self.N = N
        self.K = K

        self.pulse = np.random.normal(0, 1, N)

        self.pulse -= np.mean(self.pulse)

        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega):
        """
        Fonction définissant le modèle de Kuramoto.

        Paramètres :
        - t (float) : Temps.
        - omega (ndarray) : Tableau de fréquences angulaires.

        Retourne :
        - omega_dot (ndarray) : Tableau des dérivées temporelles des fréquences angulaires.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Résoudre le modèle de Kuramoto.

        Paramètres :
        - tmax (float) : Temps maximum.
        - step (int) : Nombre de pas de temps.

        Retourne :
        - sol (scipy.integrate.OdeSolution) : Solution de l'équation différentielle.
        """
        t = np.linspace(0, tmax, step)

        sol = solve_ivp(fun=self.KURA, t_span=(
            0, tmax), y0=self.omega, t_eval=t)

        self.sol = sol
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

        return sol

    def get_abs_ordre(self):
        """
        Calcule la valeur absolue du paramètre d'ordre.

        Retourne :
        - abs_list (ndarray) : Tableau des valeurs absolues du paramètre d'ordre.
        """
        self.abs_list = np.abs(
            np.sum(np.exp(1j * self.sol.y[:, self.sol.t >= 50]), axis=0)) / self.N
        return self.abs_list


def main_compute(args):
    """
    Fonction principale de calcul.

    Paramètres :
    - args (tuple) : Tuple d'arguments.

    Retourne :
    - mean_abs_tot (ndarray) : Tableau des valeurs absolues moyennes du paramètre d'ordre.
    """
    k_list, N, Nrep, process_id = args

    abs_tot = np.empty((len(k_list), Nrep))

    progress_bar = tqdm(total=len(k_list), position=process_id,
                        desc=f"Process {process_id} | N = {N}")

    for i, k in enumerate(k_list):
        for j in range(Nrep):
            oscis = OSCI(N, k)
            sol = oscis.solve(100, 201)
            abs_tot[i, j] = np.mean(oscis.get_abs_ordre())
        progress_bar.update(1)

    progress_bar.close()

    mean_abs_tot = np.mean(abs_tot, axis=1)
    return mean_abs_tot


if __name__ == '__main__':
    # Initialisation des paramètres
    num_proc = 5
    start_time = time.time()

    k_list = np.linspace(1.4, 1.8, 25)
    N_list = [100, 500, 2000, 5000, 15000]
    Nrep_list = [200, 50, 20, 15, 10]

    # Calcul des moyennes
    print("Starting pool")
    print("---------------------------------------------------------------------------")

    inputs = [(k_list, N_list[i], Nrep_list[i], i) for i in range(len(N_list))]
    outputs = Parallel(n_jobs=num_proc)(
        delayed(main_compute)(inp) for inp in inputs)

    print("---------------------------------------------------------------------------")
    print(f"Pool finished in {time.time() - start_time:.2f} seconds")

    # Théorie : Limite thermodynamique
    g_prime_prime = -1/np.sqrt(2*np.pi)
    g_prime = 0
    g = 1/np.sqrt(2*np.pi)
    k_c = 2/(np.pi * g)

    def r_approx(x): return np.sqrt(-16*(x - k_c) /
                                    (np.pi*(k_c**4)*g_prime_prime))

    def r(x):
        return x*(np.sqrt(2*np.pi)/4.0)*np.exp(-x**2/4.0)*(sp.iv(0, x**2/4.0) + sp.iv(1, x**2/4.0))

    y = np.linspace(0.01, 1, 120)
    k = y/r(y)

    # Graphique
    for i, result in enumerate(outputs):
        plt.plot(k_list, result, label=f"$N = {N_list[i]}$", marker='o')

    plt.axvline(x=k_c, label=r'$K_c$', color='r',
                linestyle='--', linewidth=2, alpha=0.5)
    plt.plot(k, r(y), label='Théorie Exacte', color='g',
             linestyle='--', linewidth=2, alpha=0.5)
    plt.plot(k, r_approx(k), label='Théorie Approximation',
             color='k', linestyle='--', linewidth=2, alpha=0.5)
    plt.xlabel('K')
    plt.ylabel(r'$|r(K)|$')
    plt.legend()
    plt.title(r'Moyenne de $|r(K)|$ en fonction de $K$')
    plt.tight_layout()
    plt.savefig('kura9.pdf')
    plt.show()
