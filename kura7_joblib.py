import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from tqdm import tqdm

np.random.seed(40)


class OSCI:
    def __init__(self, N, K):
        self.N = N
        self.K = K

        self.pulse = np.random.normal(0, 1, N)
        self.pulse -= np.mean(self.pulse)

        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega):
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.imag(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        t = np.linspace(0, tmax, step)

        sol = solve_ivp(fun=self.KURA, t_span=(
            0, tmax), y0=self.omega, t_eval=t)

        self.sol = sol
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

        return sol

    def get_abs_ordre(self):
        self.abs_list = np.abs(
            np.sum(np.exp(1j * self.sol.y[:, self.sol.t >= 50]), axis=0)) / self.N
        return self.abs_list


def main_compute(args):
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

    inputs = [(k_list, N_list[i], Nrep_list[i], i) for i in range(len(N_list))]
    outputs = Parallel(n_jobs=num_proc)(
        delayed(main_compute)(inp) for inp in inputs)

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