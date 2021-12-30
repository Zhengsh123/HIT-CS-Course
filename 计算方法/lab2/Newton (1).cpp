// ConsoleApplication1.cpp : 此文件包�? "main" 函数。程序执行将在�?��?�开始并结束�?
//

#include <iostream>
#include <cmath>
using namespace std;
void Legendre(int n, double x, double & y, double & dy)
{

    double polylst,poly, polyn, pdern, pderlst, pder;
    int k;
    if (n == 0)
    {
        y = 1;
        dy = 0;
        return;
    }

    if (n == 1)
    {
        y = x;
        dy = 1;
        return;
    }
    
    polylst = 1; pderlst = 0; poly = x; pder = 1;
    for ( k = 2; k <= n; k++)
    {
            polyn = ((2*k - 1) * x*poly - (k-1) * polylst) / k;
            pdern = pderlst + (2*k-1) * poly;
            polylst = poly; poly = polyn;
            pderlst = pder; pder = pdern;
    }
    y=polyn; dy =pdern;
    return;


}
double fun(double x)
{
    double  y = 0.0;
    y = (-pow(x,5)+25*pow(x,4)-200*pow(x,3)+600*pow(x,2)-600*x+120)/120;
    return y;

}
double dfun(double x)
{
    double  y = 0.0;
    y = (-5*pow(x,4)+100*pow(x,3)-600*pow(x,2)+1200*x-600)/120;
    return y;

}
double NewtonLegRoot(int n, double x0, double Eps1, double Eps2, int N)
{
    double x = 0.0;
    int i = 0;
    double F, DF, x1;
    x = x0;
    for (i = 1; i <= N; i++) {
        F = fun(x0);
        DF = dfun(x0);
        Legendre(n, x0, F, DF);
        if (abs(F) < Eps1)
        {
            x = x0;
            break;
        }
        if (abs(DF) < Eps2)
        {
            cout << "abs(DF)<Eps2 \n";
            break;
        }

        x1 = x0 - F / DF;
        if (abs(x1 - x0) < Eps1)
        {
            x = x1;
            break;
        }
        x0 = x1;
    }
    return x;
}

double Newton(double x0, double Eps1, double Eps2, int N)
{
    double x = 0.0;
    int i = 0;
    double F, DF, x1;
    x = x0;
    for (i = 1; i <= N; i++) {
        F = fun(x0);
        DF = dfun(x0);
        if (abs(F) < Eps1)
        {
            x = x0;
            break;
        }
        if (abs(DF) < Eps2)
        {
            cout << "abs(DF)<Eps2 \n";
            break;
        }

        x1 = x0 - F / DF;
        if (abs(x1 - x0) < Eps1)
        {
            x = x1;
            break;
        }
        x0 = x1;
    }
    return x;
}
    
   
int main()
{

    double Eps1 = 1.0e-6;
    double Eps2 = 1.0e-6;
    int N = 100;
    double mathpi = 3.141592653;
    // double x0[20];
    // for(int i=0;i<20;i++)
    // {
    //     x0[i]=cos((2*i+1)*mathpi/42);
    // }
    double x;
    int n = 6;
    double x0 = 12.64;
    for(int i=0;i<20;i++)
    {
         x = Newton(x0, Eps1, Eps2, N);
         cout<<x<<endl;
    }
   
    // x0 = 0.2;
    // x = NewtonLegRoot(n, x0, Eps1, Eps2, N);
    cout << x << endl;
    system("pause");
    return 0;

}
