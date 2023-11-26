import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(40)


class OSCI:
    """
    A class representing a system of oscillators.

    Attributes
    ----------
    N : int
        The number of oscillators in the system.
    K : float
        The coupling strength between oscillators.
    t_n : float
        The current time of the system.
    pulse : numpy.ndarray
        An array of random values representing the initial pulse of each oscillator.
    omega : numpy.ndarray
        An array of random values representing the initial frequency of each oscillator.
    ordre : complex
        The order parameter of the system, calculated as the average of the complex exponential of the oscillator frequencies.

    Methods
    -------
    KURA(t, omega)
        Calculates the derivative of the oscillator frequencies at a given time t and frequency omega.
    solve(tmax, step)
        Solves the system using the KURA function and the Runge-Kutta 45 method.
    graph()
        Plots the current state of the system on a complex plane and saves the plot as a PDF file.
    """

    def __init__(self, N, K):
        """
        Initializes the OSCI class.

        Parameters
        ----------
        N : int
            The number of oscillators in the system.
        K : float
            The coupling strength between oscillators.
        """
        self.N = N
        self.K = K
        self.t_n = 0
        self.pulse = np.random.normal(0, 1, N)
        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega):
        """
        Calculates the derivative of the oscillator frequencies at a given time t and frequency omega.

        Parameters
        ----------
        t : float
            The current time of the system.
        omega : numpy.ndarray
            An array of values representing the current frequency of each oscillator.

        Returns
        -------
        numpy.ndarray
            An array of values representing the derivative of the frequency of each oscillator.
        """
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot

    def solve(self, tmax, step):
        """
        Solves the system using the KURA function and the Runge-Kutta 45 method.

        Parameters
        ----------
        tmax : float
            The maximum time to solve the system for.
        step : int
            The number of steps to use in the solver.

        Returns
        -------
        scipy.integrate.OdeResult
            An object containing the solution of the system.
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
        Plots the current state of the system on a complex plane and saves the plot as a PDF file.
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
        ax.set_title(f'K = {self.K} and t = {self.t_n}')
        # plt.savefig(f'kura2_{self.N}.pdf')
        plt.show()


oscis = OSCI(100, 8)
sol = oscis.solve(100, 201)
oscis.graph()
