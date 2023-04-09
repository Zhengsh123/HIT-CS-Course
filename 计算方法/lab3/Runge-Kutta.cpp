#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;
double fun(double x,double y)
{
    double  fun_Val;
    //fun_Val = -20*(y-exp(x)*sin(x))+exp(x)*(sin(x)+cos(x));
    //fun_Val=-pow(y,2);
	fun_Val=(2*y/x)+pow(x,2)*exp(x);
	//fun_Val=(pow(y,2)+y)/x;
	//fun_Val=-20*(y-pow(x,2))+2*x;
	//fun_Val=-20*y+20*sin(x)+cos(x);
	//fun_Val=x+y;
	return fun_Val;

}
double funExactSolution(double x)
{
    double  y;
    //y = 1/(x+1);
	y=pow(x,2)*(exp(x)-exp(1));
	//y=2*x/(1-2*x);
	//y=pow(x,2)+exp(-20*x)/3;
	//y=exp(-20*x)+sin(x);
	//y=exp(x)*sin(x);
	//y=-(x+1);
    return y;

}
void RungeKutta( double x0, double T, double y0,int N,double *x,double *y)
{

	double h;
	double K1,K2,K3,K4;
	h=(T-x0)/N;
	y[0]=y0;
	x[0]=x0;

	for (int n=0;n<N;n++)
	{
		x[n+1]=x[n]+h;
		K1=fun(x[n],y[n]);
		K2=fun(x[n]+h/2.0,y[n]+h*K1/2.0);
		K3=fun(x[n]+h/2.0,y[n]+h*K2/2.0);
		K4=fun(x[n]+h,y[n]+h*K3);
		y[n+1]=y[n]+1.0/6*h*(K1+2*K2+2*K3+K4);


	}
    return;
}


int main()
{
    double *x=NULL ;
    double *y=NULL;
    double *ExactSolutiony=NULL;
    double x0;
	double T;
	double y0;
	int N;
    N=10;
    x= new double [N+1];
    y = new double [N+1];
    ExactSolutiony = new double [N+1];
    x0=1;
    y0=0;
    T=3;
    RungeKutta(  x0,  T,  y0, N,x,y);
    for (int n=0;n<=N;n++)
    {
    	 ExactSolutiony[n]=funExactSolution( x[n]);
    	//  cout <<"x=";
		//cout<<setw(4) << setprecision(3) << x[n] <<endl;
		//  cout <<"y=";
		//cout<<setw(4) << setprecision(4) << y[n] <<endl;
		//  cout <<"ExactSolutiony=";
		//  cout<<setw(9) << setprecision(6) << ExactSolutiony[n] <<"   ";
		//  cout<<"error=";
		cout<<setw(1) << setprecision(1) << ExactSolutiony[n]-y[n] <<endl;
	}



	delete [] x;
	delete [] y;
	delete [] ExactSolutiony;
	system("pause");
	return 0;
}

