// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <cmath>
#include <iomanip>
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
            polyn = 2*x*poly - 2*(k-1)*polylst;
            pdern = 2*x*pder +2* poly -2*(k-1)*pderlst ;
            polylst = poly; poly = polyn;
            pderlst = pder; pder = pdern;
    }
    y=polyn; dy =pdern;
    return;


}
double fun(double x)
{
    double  y = 0.0;
    y = cos(x) - x;
    return y;

}
double dfun(double x)
{
    double  y = 0.0;
    y = -sin(x) - 1.0;
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
    double Eps2 = 1.0e-4;
    int N = 10000;
    double mathpi = 3.141592653589793238463;
    double x0;
    double x;
    int n = 6;
    x0 = mathpi / 4;
    cout << x0 << endl;
    //x = Newton(x0, Eps1, Eps2, N);
    x0 = 1.3;
    x = NewtonLegRoot(n, x0, Eps1, Eps2, N);
    cout <<setw(10) << setprecision(8) << x << endl;

}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧:
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
