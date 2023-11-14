# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:00:38 2023

@author: tradu
"""

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
        self.pulse[-1] = -np.sum(self.pulse[:-1])
        print(np.sum(self.pulse))
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
        omega_dot = self.pulse + self.K * \
            np.abs(ordre) * np.sin(np.imag(ordre) - omega)
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

        sol = solve_ivp(fun=self.KURA, t_span=(
            self.t_n, tmax), y0=self.omega, t_eval=t)

        abs_list = np.empty(step, dtype=np.cdouble)
        print(sol.y.shape)
        for i in range(step):
            tmp = sol.y[:, i]
            abs_list[i]=np.sum(np.exp(1j * tmp)) / self.N
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += t[-1]

        return abs_list,t

K = np.linspace(1, 2, 6)
for i in K:
    oscis = OSCI(100, i)
    abs_ordre, t = oscis.solve(100, 201)
    plt.plot(t, abs_ordre, label = str(i))
    
plt.legend()
plt.savefig('kura4_100.pdf')