/**
 * This file contains the implementation of the FPUT class and its main
 * function.
 */

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

typedef struct {
  double x, v;
} double2;

/**
 * The FPUT class represents the Fermi-Pasta-Ulam-Tsingou (FPUT) system.
 */
class FPUT {
  int N;                /**< The number of particles in the system. */
  double alpha;         /**< The alpha parameter of the system. */
  double A;             /**< The A parameter of the system. */
  vector<double2> par;  /**< The positions and velocities of the particles. */
  vector<double> omega; /**< The angular frequencies of the particles. */
  vector<vector<double>>
      A_mat; /**< The matrix used in the calculation of energies. */

public:
  /**
   * Constructs a new FPUT object.
   *
   * n The number of particles in the system.
   * a The A parameter of the system.
   * al The alpha parameter of the system.
   */
  FPUT(int n, double a, double al) {
    N = n;
    A = a;
    alpha = al;
    par.resize(N + 1);
    omega.resize(N + 1);
    A_mat.resize(N + 1, vector<double>(N + 1));

    for (int i = 0; i <= N; i++) {
      par[i].x = A * sin(i * M_PI / N);
      par[i].v = 0.0;

      omega[i] = 2.0 * sin(i * M_PI / (2.0 * N));

      for (int j = 0; j <= N; j++) {
        A_mat[i][j] = sqrt(2.0 / N) * sin(i * j * M_PI / N);
      }
    }
  }

  /**
   * Calculates the total energy of the system.
   *
   * return The total energy of the system.
   */
  double energie_tot() {
    double E = 0.0;
    for (int i = 1; i < N; i++) {
      E += par[i].v * par[i].v / 2.0;
    }
    for (int i = 0; i < N; i++) {
      E += pow(par[i + 1].x - par[i].x, 2) / 2.0 +
           alpha * pow(par[i + 1].x - par[i].x, 3) / 3.0;
    }
    return E;
  }

  /**
   * Calculates the energies of the system's modes.
   *
   * return A vector containing the energies of the system's modes.
   */
  vector<double> energie_modes() {
    vector<double> Q(N + 1);
    vector<double> Q_dot(N + 1);
    vector<double> Ek(N + 1);

    for (int k = 0; k <= N; k++) {
      Q[k] = 0.0;
      Q_dot[k] = 0.0;
      Ek[k] = 0.0;

      for (int j = 0; j <= N; j++) {
        Q[k] += par[j].x * A_mat[k][j];
        Q_dot[k] += par[j].v * A_mat[k][j];
      }

      Ek[k] += pow(Q_dot[k], 2) / 2.0 + pow(omega[k], 2) * pow(Q[k], 2) / 2.0;
    }

    return Ek;
  }
};

/**
 * The main function.
 *
 * return The exit status of the program.
 */
int main() {
  FPUT fput1(32, 1.0, 0.0);
  cout << "FPUT 1:" << endl;
  vector<double> Ek = fput1.energie_modes();

  for (int i = 0; i < 4; i++) {
    cout << Ek[i] << endl;
  }

  FPUT fput2(32, 1.0, 0.25);
  cout << "FPUT 2:" << endl;
  vector<double> Ek2 = fput2.energie_modes();

  for (int i = 0; i < 4; i++) {
    cout << Ek2[i] << endl;
  }

  return 0;
}