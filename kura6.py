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

        abs_list = np.empty(step)

        # Enlever cette boucle et remplacer par express vectorise
        for i in range(step):
            tmp = sol.y[:, i]
            abs_list[i]=np.abs(np.sum(np.exp(1j * tmp))) / self.N
        self.omega = sol.y[:, -1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += t[-1]

        return abs_list,t

K = np.linspace(1.4, 1.8, 25)
N=100
Nrep=10
mean_ordre = np.empty((len(K), Nrep))

for i in range(len(K)):
    for j in range(Nrep):
        oscis = OSCI(N, K[i])
        abs_ordre, t = oscis.solve(N, 201)
        mean_ordre[i, j] = np.mean(abs_ordre[t >= 50])
        
mean_mean_ordre = np.mean(mean_ordre, axis=1)  
    
plt.plot(K,mean_mean_ordre)
plt.savefig(f'kura5_{Nrep}.pdf')