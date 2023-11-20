/**
 * file fput1.cpp
 * Implementation of the FPUT class and its main function.
 */

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

typedef struct {
  double x, v;
} double2;

/**
 * class FPUT
 * Class representing the Fermi-Pasta-Ulam-Tsingou (FPUT) system.
 */
class FPUT {
  int N;               /**< Number of particles */
  double alpha;        /**< Nonlinear coefficient */
  double A;            /**< Amplitude */
  vector<double2> par; /**< Vector of particle positions and velocities */

public:
  /**
   * Constructor for the FPUT class.
   * param n Number of particles
   * param a Amplitude
   * param al Nonlinear coefficient
   */
  FPUT(int n, double a, double al) {
    N = n;
    A = a;
    alpha = al;
    par.resize(N + 1);

    for (int i = 0; i <= N; i++) {
      par[i].x = A * sin(i * M_PI / N);
      par[i].v = 0.0;
    }
  }

  /**
   * Calculate the total energy of the FPUT system.
   * return Total energy of the system
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
};

/**
 * Main function.
 * return Exit status
 */
int main() {
  FPUT fput1(32, 1.0, 0.0);
  cout << fput1.energie_tot() << endl;

  FPUT fput2(32, 1.0, 0.25);
  cout << fput2.energie_tot() << endl;

  return 0;
}