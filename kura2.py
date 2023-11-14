import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(40)

class OSCI :
    def __init__(self, N, K) :
        self.N = N
        self.K = K
        self.t_n = 0
        self.pulse = np.random.normal(0, 1, N)
        self.omega = np.random.uniform(-np.pi, np.pi, N)
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N

    def KURA(self, t, omega) :
        ordre = np.sum(np.exp(1j * omega)) / self.N
        omega_dot = self.pulse + self.K * np.abs(ordre) * np.sin(np.angle(ordre) - omega)
        return omega_dot
        
    
    def solve(self, tmax, step) :
        # Solve using RK45 from solve_ivp
        t = np.linspace(self.t_n, tmax, step)
        sol = solve_ivp(fun = self.KURA, t_span=(self.t_n, tmax), y0 = self.omega, t_eval=t)
        self.omega = sol.y[:,-1]
        self.ordre = np.sum(np.exp(1j * self.omega)) / self.N
        self.t_n += tmax

        return sol # Sert à rien mais bon


    def graph(self) :
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
        plt.savefig(f'kura2_{self.N}.pdf')
        plt.show()

oscis = OSCI(100, 0)
sol = oscis.solve(100, 201)
oscis.graph()
