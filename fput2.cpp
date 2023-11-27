/**
 * Ce fichier contient l'implémentation de la classe FPUT et de sa fonction principale.
 */

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

typedef struct {
  double x, v;
} double2;

/**
 * La classe FPUT représente le système Fermi-Pasta-Ulam-Tsingou (FPUT).
 */
class FPUT {
  int N;                /**< Le nombre de particules dans le système. */
  double alpha;         /**< Le paramètre alpha du système. */
  double A;             /**< Le paramètre A du système. */
  vector<double2> par;  /**< Les positions et vitesses des particules. */
  vector<double> omega; /**< Les fréquences angulaires des particules. */
  vector<vector<double>>
      A_mat; /**< La matrice utilisée dans le calcul des énergies. */

public:
  /**
   * Construit un nouvel objet FPUT.
   *
   * n Le nombre de particules dans le système.
   * a Le paramètre A du système.
   * al Le paramètre alpha du système.
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
   * Calcule l'énergie totale du système.
   *
   * return L'énergie totale du système.
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
   * Calcule les énergies des modes du système.
   *
   * return Un vecteur contenant les énergies des modes du système.
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
 * La fonction principale.
 *
 * return Le code de sortie du programme.
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