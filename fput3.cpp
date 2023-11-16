#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

typedef struct
{
    double x, v;
} double2;

class FPUT
{
public:
    int N;
    double alpha = 0.25;
    vector<double2> par;
    vector<double2> parn; // next time step
    vector<double> Ek;
    vector<double> omega;
    vector<vector<double>> AA;
    double nrj = 0;
    double ampli = 0;
    double pi = acos(-1);
    double Deltat = 0.01;
    FPUT(int n, double Alpha, double ampli, double Deltat) : N(n), alpha(Alpha), ampli(ampli), Deltat(Deltat)
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

    double calculE()
    {
        double E = 0.0;
        for (int i = 1; i < N; i++)
        {
            E += par[i].v * par[i].v / 2.0;
        }
        for (int i = 0; i < N; i++)
        {
            E += pow(par[i + 1].x - par[i].x, 2) / 2.0 + alpha * pow(par[i + 1].x - par[i].x, 3) / 3.0;
        }
        return E;
    }

    vector<double> calculEk()
    {
        vector<double> Q(N + 1);
        vector<double> Q_dot(N + 1);
        vector<double> Ek(N + 1);

        for (int k = 0; k <= N; k++)
        {
            Q[k] = 0.0;
            Q_dot[k] = 0.0;
            Ek[k] = 0.0;

            for (int j = 0; j <= N; j++)
            {
                Q[k] += par[j].x * AA[k][j];
                Q_dot[k] += par[j].v * AA[k][j];

                Ek[k] += pow(Q_dot[k], 2) / 2.0 + omega[k] * omega[k] * pow(Q[k], 2) / 2.0;
            }
        }

        return Ek;
    }

    double force(double x)
    {
        return (-x - alpha * x * x);
    }

    void verlet()
    {
        double Deltat2 = Deltat * Deltat;
        for (int i = 1; i < N; ++i)
        {
            double tmp = -force(par[i + 1].x - par[i].x) + force(par[i].x - par[i - 1].x);
            parn[i].x = par[i].x + Deltat * par[i].v + 0.5 * Deltat2 * tmp;
        }
        for (int i = 1; i < N; ++i)
        {
            double tmp = -force(par[i + 1].x - par[i].x) + force(par[i].x - par[i - 1].x);
            double tmp2 = -force(parn[i + 1].x - parn[i].x) + force(parn[i].x - parn[i - 1].x);
            parn[i].v = par[i].v + 0.5 * Deltat * (tmp + tmp2);
        }
        for (int i = 0; i < N + 1; i++)
        {
            par[i].x = parn[i].x;
            par[i].v = parn[i].v;
        }
    }
};

int main()
{
    cout << "Starting Simulation" << endl;
    int N = 32;
    int Nsteps = 100000;
    double pi = acos(-1);
    double DeltaT = 0.01;
    FPUT fput(N, 0.25, 1.0, DeltaT);

    //Write to energies_alpha.dat
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
            file << i << "\t" << fput.nrj << "\t" << fput.Ek[0] << "\t" << fput.Ek[1] << "\t" << fput.Ek[2] << "\t" << fput.Ek[3] << endl;
        }
    }

    cout << "Simulation done" << endl;

    file.close();
    return (0);
}