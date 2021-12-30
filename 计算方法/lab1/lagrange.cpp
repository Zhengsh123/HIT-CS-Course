#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
using namespace std;
#define PI acos(-1)
double Lagrange(int n, double *x, double *funx, double Newx)
{
    double y=0.0;
    int k=0;
    while(k<=n)
    {
        double l=1.0;
        for(int j=0;j<=n;j++)
        {
            if(j!=k)
            {
                l=l*(Newx-x[j])/(x[k]-x[j]);
            }
        }
        y+=l*funx[k];
        k++;
    }
    return y;
}
double fun1(double x)
{
    double  y = 0.0;
    y = 1/(1+x*x);
    return y;
}
double fun2(double x)
{
    return exp(x);
}
double fun3(double x)
{
    return sqrt(x);
}
int main()
{
    double *x=NULL ;
    double *funx=NULL;
    int n,k ;
    double Newx[5]={5,50,115,185};
    double Newy[5]={0};
    double h;
    int Newx_Length=4;
    ofstream outputfile;
    n=5;
    x= new double [n+1];
    funx = new double [n+1];
    h=2/n;
    // for (k=0;k<=n;k++)
    // {
    //   	x[k]=cos((2*k+1)*PI/(2*(n+1)));
    //   	funx[k]=fun2(x[k]);
	// } 
    x[0]=169;
    x[1]=196;
    x[2]=225;
    for(int i=0;i<3;i++)
    {
        funx[i]=fun3(x[i]);
        cout<<funx[i]<<endl;
    }
    for (int i=0;i<Newx_Length;i++)
    {
    	 Newy[i]=Lagrange(2, x, funx,Newx[i]);
    	 cout <<"x="; 
		 cout<<setw(4) << setprecision(3) << Newx[i] <<"   ";
		 cout <<"y=";
		 cout<<setw(9) << setprecision(6) << Newy[i] <<endl;
	}
	
	
	
    outputfile.open ("Lagrange.txt");
    
    if (outputfile.is_open()) 
	{    	    
      for (int i=0;i<Newx_Length;i++)
      {   	 
	       outputfile <<"x="; 
		   outputfile<<setw(4)  << setprecision(3) << Newx[i] <<"   ";
		   outputfile <<"y=";
		   outputfile<<setw(9) << setprecision(6) << Newy[i] <<endl;
	  }    	
    } 
    outputfile.close();	
	delete [] x;
	delete [] funx;
    system("pause"); 
	return 0;

}