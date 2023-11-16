#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

typedef struct
{
    double x, v;
} double2;

class FPUT
{
    int N;
    double alpha, A;
    vector<double2> par;
    vector<double> omega;
    vector<vector<double>> A_mat;

public:
    FPUT(int n, double a, double al)
    {
        N = n;
        A = a;
        alpha = al;
        par.resize(N + 1);
        omega.resize(N + 1);
        A_mat.resize(N + 1, vector<double>(N + 1));

        for (int i = 0; i <= N; i++)
        {
            par[i].x = A * sin(i * M_PI / N);
            par[i].v = 0.0;

            omega[i] = 2.0 * sin(i * M_PI / (2.0 * N));

            for (int j = 0; j <= N; j++)
            {
                A_mat[i][j] = sqrt(2.0 / N) * sin(i * j * M_PI / N);
            }
        }
    }

    double energie_tot()
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

    vector<double> energie_modes()
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
                Q[k] += par[j].x * A_mat[k][j];
                Q_dot[k] += par[j].v * A_mat[k][j];

                Ek[k] += pow(Q_dot[k], 2) / 2.0 + omega[k] * omega[k] * pow(Q[k], 2) / 2.0;
            }
        }

        return Ek;
    }
};

int main()
{
    FPUT fput1(32, 1.0, 0.0);
    cout << "FPUT 1:" << endl;

    for (int i = 0; i < 4; i++)
    {
        cout << fput1.energie_modes()[i] << endl;
    }
    

    FPUT fput2(32, 1.0, 0.25);
    cout << "FPUT 2:" << endl;

    for (int i = 0; i < 4; i++)
    {
        cout << fput2.energie_modes()[i] << endl;
    }

    return 0;
}