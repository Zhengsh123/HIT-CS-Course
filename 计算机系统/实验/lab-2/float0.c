#include <stdio.h>
#include<stdlib.h>
#include<float.h>
#include <math.h>
int main()
{
    float c = FLT_MIN;
    float a=0;
    float b=1/a;
    printf("浮点数除以0：%f\n",b);
    printf("浮点数除以极小的浮点数：%f\n",1/c);
    system("pause");
    return 0;
}