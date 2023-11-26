import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multiprocessing import Pool
from tqdm import tqdm

np.random.seed(40)

# ATTENTION LA PARALLELISATION NE MARCHE PAS
# IL FAUT UTILISER joblib


class OSCI:
    """
    Classe représentant un Système d'Oscillateurs avec Interaction Collective (OSCI).

    Attributs:
        N (int): Nombre d'oscillateurs dans le système.
        K (float): Force de couplage entre les oscillateurs.

    Méthodes:
        __init__(self, N, K): Initialise l'objet OSCI avec les paramètres donnés.
        KURA(self, t, omega): Calcule la dérivée des phases des oscillateurs.
        solve(self, tmax, step): Résout le système OSCI en utilisant une intégration numérique.
        graph(self): Trace la distribution des phases et le paramètre d'ordre.
        get_abs_ordre(self): Calcule la valeur absolue du paramètre d'ordre au fil du temps.
    """

    def __init__(self, N, K):
        self.N = N
        self.K = K

        self.pulse = np.random.normal(0, 1, N)
        self.pulse -= np.mean(self.pulse)

        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega):
        """
        Calcule la dérivée des phases des oscillateurs.

        Args:
            t (float): Temps.
            omega (ndarray): Tableau des phases des oscillateurs.

        Returns:
            ndarray: Tableau des dérivées des phases des oscillateurs.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Résout le système OSCI en utilisant une intégration numérique.

        Args:
            tmax (float): Temps maximum.
            step (int): Nombre d'étapes temporelles.

        Returns:
            OdeSolution: Solution du système OSCI.
        """
        t = np.linspace(0, tmax, step)

        sol = solve_ivp(fun=self.KURA, t_span=(
            0, tmax), y0=self.omega, t_eval=t)

        self.sol = sol
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

        return sol

    def graph(self):
        """
        Trace la distribution des phases et le paramètre d'ordre.
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
        """
        Calcule la valeur absolue du paramètre d'ordre au fil du temps.

        Returns:
            ndarray: Tableau des valeurs absolues du paramètre d'ordre.
        """
        self.abs_list = np.abs(
            np.sum(np.exp(1j * self.sol.y[:, self.sol.t >= 50]), axis=0)) / self.N
        return self.abs_list


def main_compute(args):
    """
    Calcule la valeur moyenne absolue du paramètre d'ordre pour différentes valeurs de K et N.

    Args:
        args (tuple): Tuple contenant la liste des valeurs de K, la valeur de N, le nombre de répétitions et l'ID du processus.

    Returns:
        ndarray: Tableau des valeurs moyennes absolues du paramètre d'ordre.
    """
    k_list, N, Nrep, process_id = args

    abs_tot = np.empty((len(k_list), Nrep))

    progress_bar = tqdm(total=len(k_list), position=process_id,
                        desc=f"Processus {process_id} | N = {N}")

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
    num_proc = 5

    k_list = np.linspace(1.4, 1.8, 25)
    N_list = [100, 500, 2000, 5000, 15000]
    Nrep_list = [200, 50, 20, 15, 10]

    print("Démarrage du pool")
    print("---------------------------------------------------------------------------")

    with Pool(num_proc) as pool:
        inputs = [(k_list, N_list[i], Nrep_list[i], i)
                  for i in range(len(N_list))]
        outputs = pool.map(main_compute, inputs)

    print("---------------------------------------------------------------------------")
    print("Pool terminé")

    for i, result in enumerate(outputs):
        plt.plot(k_list, result, label=f"N = {N_list[i]}", marker='o')
    plt.xlabel('K')
    plt.ylabel('abs(ordre)')
    plt.legend()
    plt.title('Moyenne de abs(ordre) en fonction de K')
    plt.tight_layout()
    plt.savefig('kura7.pdf')
    plt.show()
