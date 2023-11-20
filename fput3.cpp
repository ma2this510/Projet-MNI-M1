/**
 * fput3.cpp
 * Implementation of the FPUT class and main function for simulating the Fermi-Pasta-Ulam-Tsingou (FPUT) system.
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

typedef struct
{
  double x, v;
} double2;

/**
 * FPUT
 * Class representing the Fermi-Pasta-Ulam-Tsingou (FPUT) system.
 */
class FPUT
{
public:
  int N;                     /**< Number of particles */
  double alpha = 0.25;       /**< Nonlinearity parameter */
  vector<double2> par;       /**< Vector of particle positions and velocities */
  vector<double2> parn;      /**< Vector of particle positions and velocities at the next time step */
  vector<double> Ek;         /**< Vector of kinetic energies */
  vector<double> omega;      /**< Vector of angular frequencies */
  vector<vector<double>> AA; /**< Matrix of coefficients */
  double nrj = 0;            /**< Total energy of the system */
  double ampli = 0;          /**< Amplitude of the initial displacement */
  double pi = acos(-1);      /**< Value of pi */
  double Deltat = 0.01;      /**< Time step size */

  /**
   * Constructor for the FPUT class.
   * param n Number of particles
   * param Alpha Nonlinearity parameter
   * param ampli Amplitude of the initial displacement
   * param Deltat Time step size
   */
  FPUT(int n, double Alpha, double ampli, double Deltat)
      : N(n), alpha(Alpha), ampli(ampli), Deltat(Deltat)
  {
    par.resize(N + 1);
    parn.resize(N + 1);
    Ek.resize(N + 1);
    omega.resize(N + 1);
    AA.resize(N + 1, vector<double>(N + 1));

    for (int i = 0; i <= N; i++)
    {
      par[i].x = ampli * sin(i * M_PI / N);
      par[i].v = 0.0;

      omega[i] = 2.0 * sin(i * M_PI / (2.0 * N));

      for (int j = 0; j <= N; j++)
      {
        AA[i][j] = sqrt(2.0 / N) * sin(i * j * M_PI / N);
      }
    }
  }

  /**
   * Calculates the total energy of the system.
   * return Total energy of the system
   */
  double calculE()
  {
    double E = 0.0;
    for (int i = 1; i < N; i++)
    {
      E += par[i].v * par[i].v / 2.0;
    }
    for (int i = 0; i < N; i++)
    {
      E += pow(par[i + 1].x - par[i].x, 2) / 2.0 +
           alpha * pow(par[i + 1].x - par[i].x, 3) / 3.0;
    }
    return E;
  }

  /**
   * Calculates the kinetic energies of the particles.
   * return Vector of kinetic energies
   */
  vector<double> calculEk()
  {
    vector<double> Q(N + 1);
    vector<double> Q_dot(N + 1);
    vector<double> Ek(N + 1);

    for (int k = 0; k <= N; k++)
    {
      Q[k] = 0.0;
      Q_dot[k] = 0.0;

      for (int j = 1; j < N; j++)
      {
        Q[k] += par[j].x * AA[k][j];
        Q_dot[k] += par[j].v * AA[k][j];
      }

      Ek[k] = pow(Q_dot[k], 2) / 2.0 + omega[k] * omega[k] * pow(Q[k], 2) / 2.0;
    }

    return Ek;
  }

  /**
   * Calculates the force acting on a particle.
   * param x Position of the particle
   * return Force acting on the particle
   */
  double force(double x) { return (-x - alpha * x * x); }

  /**
   * @brief Performs the Verlet integration to update the particle positions and velocities.
   */
  void verlet()
  {
    double Deltat2 = Deltat * Deltat;
    for (int i = 1; i < N; ++i)
    {
      double tmp =
          -force(par[i + 1].x - par[i].x) + force(par[i].x - par[i - 1].x);
      parn[i].x = par[i].x + Deltat * par[i].v + 0.5 * Deltat2 * tmp;
    }
    for (int i = 1; i < N; ++i)
    {
      double tmp =
          -force(par[i + 1].x - par[i].x) + force(par[i].x - par[i - 1].x);
      double tmp2 =
          -force(parn[i + 1].x - parn[i].x) + force(parn[i].x - parn[i - 1].x);
      parn[i].v = par[i].v + 0.5 * Deltat * (tmp + tmp2);
    }
    for (int i = 0; i < N + 1; i++)
    {
      par[i].x = parn[i].x;
      par[i].v = parn[i].v;
    }
  }
};

/**
 * Main function for running the simulation.
 * return Exit status of the program
 */
int main()
{
  cout << "Starting Simulation" << endl;
  int N = 32;           /**< Number of particles */
  int Nsteps = 100000;  /**< Number of simulation steps */
  double pi = acos(-1); /**< Value of pi */
  double DeltaT = 0.1;  /**< Time step size */
  FPUT fput(N, 0.25, 1.0, DeltaT);

  // Write to energies_alpha.dat
  ofstream file("energies_alpha.dat");

  for (int i = 0; i < Nsteps; i++)
  {
    fput.verlet();
    // Needed ?
    fput.nrj = fput.calculE();
    fput.Ek = fput.calculEk();

    if (i % 100 == 0)
    {
      cout << "Step " << i << " done" << endl;
      file << i << "\t" << fput.nrj << "\t" << fput.Ek[1] << "\t" << fput.Ek[2]
           << "\t" << fput.Ek[3] << "\t" << fput.Ek[4] << endl;
    }
  }

  cout << "Simulation done" << endl;

  file.close();
  return (0);
}