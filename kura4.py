import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(40)


class OSCI:
    def __init__(self, N, K):
        """
        Initializes the OSCI class with the given parameters.

        Parameters:
        N (int): The number of oscillators.
        K (float): The coupling strength.
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
        Calculates the derivative of the phase angles of the oscillators.

        Parameters:
        t (float): The current time.
        omega (ndarray): The current phase angles of the oscillators.

        Returns:
        ndarray: The derivative of the phase angles of the oscillators.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        self.abs_list = np.append(self.abs_list, np.abs(ordre))
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Solves the differential equation for the given time range.

        Parameters:
        tmax (float): The maximum time to solve for.
        step (int): The number of time steps to use.

        Returns:
        OdeResult: The solution of the differential equation.
        """
        t = np.linspace(self.t_n, tmax, step)

        self.abs_list = np.empty(0)
        sol = solve_ivp(fun=self.KURA, t_span=(
            self.t_n, tmax), y0=self.omega, t_eval=t)
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += t[-1]

        return sol

    def graph(self):
        """
        Plots the current state of the oscillators.

        Creates a scatter plot of the current state of the oscillators, with the x-axis representing the cosine of the oscillator's phase and the y-axis representing the sine of the oscillator's phase. The plot also includes a circle with radius 1 centered at the origin, which represents the unit circle. The current order parameter is plotted as a green dot, and the current value of K and t_n are included in the plot title. The resulting plot is saved as a PDF file and displayed.
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
        axs[0].set_xlabel('$\cos$ and $real$')
        axs[0].set_ylabel('$\sin$ and $imag$')
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
        return self.abs_list


for k in np.linspace(1, 2, 6):
    oscis = OSCI(100, k)
    sol = oscis.solve(100, 201)
    tmp = oscis.get_abs_ordre()
    plt.plot(np.linspace(0, 100, len(tmp)), tmp, label=f'K = {k}')

plt.xlabel('t')
plt.ylabel('abs(ordre)')
plt.legend()
plt.tight_layout()
plt.savefig('kura4.pdf')
plt.show()
