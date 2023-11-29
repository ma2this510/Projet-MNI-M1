/**
 * fput3.cpp
 * Implémentation de la classe FPUT et de la fonction principale pour simuler le
 * système Fermi-Pasta-Ulam-Tsingou (FPUT).
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

typedef struct {
  double x, v;
} double2;

/**
 * FPUT
 * Classe représentant le système Fermi-Pasta-Ulam-Tsingou (FPUT).
 */
class FPUT {
public:
  int N;                /**< Nombre de particules */
  double alpha = 0.25;  /**< Paramètre de non-linéarité */
  vector<double2> par;  /**< Vecteur des positions et vitesses des particules */
  vector<double2> parn; /**< Vecteur des positions et vitesses des particules au
                           prochain pas de temps */
  vector<double> Ek;    /**< Vecteur des énergies cinétiques */
  vector<double> omega; /**< Vecteur des fréquences angulaires */
  vector<vector<double>> AA; /**< Matrice des coefficients */
  double nrj = 0;            /**< Énergie totale du système */
  double ampli = 0;          /**< Amplitude du déplacement initial */
  double pi = acos(-1);      /**< Valeur de pi */
  double Deltat;     /**< Taille du pas de temps */

  /**
   * Constructeur de la classe FPUT.
   * param n Nombre de particules
   * param Alpha Paramètre de non-linéarité
   * param ampli Amplitude du déplacement initial
   * param Deltat Taille du pas de temps
   */
  FPUT(int n, double Alpha, double ampli, double Deltat)
      : N(n), alpha(Alpha), ampli(ampli), Deltat(Deltat) {
    par.resize(N + 1);
    parn.resize(N + 1);
    Ek.resize(N + 1);
    omega.resize(N + 1);
    AA.resize(N + 1, vector<double>(N + 1));

    for (int i = 0; i <= N; i++) {
      par[i].x = ampli * sin(i * M_PI / N);
      par[i].v = 0.0;

      omega[i] = 2.0 * sin(i * M_PI / (2.0 * N));

      for (int j = 0; j <= N; j++) {
        AA[i][j] = sqrt(2.0 / N) * sin(i * j * M_PI / N);
      }
    }
  }

  /**
   * Calcule l'énergie totale du système.
   * return Énergie totale du système
   */
  double calculE() {
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
   * Calcule les énergies cinétiques des particules.
   * return Vecteur des énergies cinétiques
   */
  vector<double> calculEk() {
    vector<double> Q(N + 1);
    vector<double> Q_dot(N + 1);
    vector<double> Ek(N + 1);

    for (int k = 0; k <= N; k++) {
      Q[k] = 0.0;
      Q_dot[k] = 0.0;

      for (int j = 1; j < N; j++) {
        Q[k] += par[j].x * AA[k][j];
        Q_dot[k] += par[j].v * AA[k][j];
      }

      Ek[k] = pow(Q_dot[k], 2) / 2.0 + omega[k] * omega[k] * pow(Q[k], 2) / 2.0;
    }

    return Ek;
  }

  /**
   * Calcule la force agissant sur une particule.
   * param x Position de la particule
   * return Force agissant sur la particule
   */
  double force(double x) { return (-x - alpha * x * x); }

  /**
   * @brief Effectue l'intégration de Verlet pour mettre à jour les positions et
   * vitesses des particules.
   */
  void verlet() {
    double Deltat2 = Deltat * Deltat;
    for (int i = 1; i < N; ++i) {
      double tmp =
          -force(par[i + 1].x - par[i].x) + force(par[i].x - par[i - 1].x);
      parn[i].x = par[i].x + Deltat * par[i].v + 0.5 * Deltat2 * tmp;
    }
    for (int i = 1; i < N; ++i) {
      double tmp =
          -force(par[i + 1].x - par[i].x) + force(par[i].x - par[i - 1].x);
      double tmp2 =
          -force(parn[i + 1].x - parn[i].x) + force(parn[i].x - parn[i - 1].x);
      parn[i].v = par[i].v + 0.5 * Deltat * (tmp + tmp2);
    }
    for (int i = 0; i < N + 1; i++) {
      par[i].x = parn[i].x;
      par[i].v = parn[i].v;
    }
  }
};

/**
 * Fonction principale pour exécuter la simulation.
 * return Code de sortie du programme
 */
int main() {
  cout << "Démarrage de la simulation" << endl;
  int N = 32;           /**< Nombre de particules */
  int Nsteps = 30*100000;  /**< Nombre d'étapes de simulation */
  double DeltaT = 0.1;  /**< Taille du pas de temps */
  FPUT fput(N, 0.25, 1.0, DeltaT);

  // Écrire dans energies_alpha.dat
  ofstream file("energies_alpha_long.dat");

  for (int i = 0; i < Nsteps; i++) {
    fput.verlet();
    fput.nrj = fput.calculE();
    fput.Ek = fput.calculEk();

    if (i % 100 == 0) {
    //   cout << "Étape " << i << " terminée" << endl;
      file << i*DeltaT << "\t" << fput.nrj << "\t" << fput.Ek[1] << "\t" << fput.Ek[2]
           << "\t" << fput.Ek[3] << "\t" << fput.Ek[4] << endl;
    }
  }

  cout << "Simulation terminée" << endl;

  file.close();
  return (0);
}