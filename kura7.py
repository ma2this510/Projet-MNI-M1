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
    Class representing an Oscillator System with Collective Interaction (OSCI).

    Attributes:
        N (int): Number of oscillators in the system.
        K (float): Coupling strength between oscillators.

    Methods:
        __init__(self, N, K): Initializes the OSCI object with the given parameters.
        KURA(self, t, omega): Calculates the derivative of the oscillator phases.
        solve(self, tmax, step): Solves the OSCI system using numerical integration.
        graph(self): Plots the phase distribution and order parameter.
        get_abs_ordre(self): Calculates the absolute value of the order parameter over time.
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
        Calculates the derivative of the oscillator phases.

        Args:
            t (float): Time.
            omega (ndarray): Array of oscillator phases.

        Returns:
            ndarray: Array of derivatives of the oscillator phases.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Solves the OSCI system using numerical integration.

        Args:
            tmax (float): Maximum time.
            step (int): Number of time steps.

        Returns:
            OdeSolution: Solution of the OSCI system.
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
        Plots the phase distribution and order parameter.
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
        axs[0].set_title(f'K = {self.K} and t = {self.t_n}')
        axs[0].set_xlabel(r'$\cos$ and $real$')
        axs[0].set_ylabel(r'$\sin$ and $imag$')
        axs[1].plot(np.linspace(0, self.t_n, len(
            self.abs_list)), self.abs_list)
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('abs(ordre)')
        axs[1].set_title(f'$r(t)$')
        axs[1].set_aspect('equal')
        plt.tight_layout()
        # plt.savefig(f'kura4_{self.N}.pdf')
        plt.show()

    def get_abs_ordre(self):
        """
        Calculates the absolute value of the order parameter over time.

        Returns:
            ndarray: Array of absolute values of the order parameter.
        """
        self.abs_list = np.abs(
            np.sum(np.exp(1j * self.sol.y[:, self.sol.t >= 50]), axis=0)) / self.N
        return self.abs_list


def main_compute(args):
    """
    Computes the mean absolute value of the order parameter for different values of K and N.

    Args:
        args (tuple): Tuple containing the list of K values, N value, number of repetitions, and process ID.

    Returns:
        ndarray: Array of mean absolute values of the order parameter.
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
    num_proc = 5

    k_list = np.linspace(1.4, 1.8, 25)
    N_list = [100, 500, 2000, 5000, 15000]
    Nrep_list = [200, 50, 20, 15, 10]

    print("Starting pool")
    print("---------------------------------------------------------------------------")

    with Pool(num_proc) as pool:
        inputs = [(k_list, N_list[i], Nrep_list[i], i)
                  for i in range(len(N_list))]
        outputs = pool.map(main_compute, inputs)

    print("---------------------------------------------------------------------------")
    print("Pool finished")

    for i, result in enumerate(outputs):
        plt.plot(k_list, result, label=f"N = {N_list[i]}", marker='o')
    plt.xlabel('K')
    plt.ylabel('abs(ordre)')
    plt.legend()
    plt.title(f'Moyenne de abs(ordre) en fonction de K')
    plt.tight_layout()
    plt.savefig(f'kura7.pdf')
    plt.show()
