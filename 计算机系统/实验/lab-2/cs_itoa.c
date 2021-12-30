#include <stdio.h>
#include<stdlib.h>
char* cs_itoa(int num);
int main()
{
    int num;
    scanf("%d",&num);
    cs_itoa(num);
    system("pause");
    return 0;
}
char* cs_itoa(int num)
{
    char res[100];
    int temp=0;
    int i=0;
    int j=1;
    if(num<0)
    {
        res[0]='-';
        i++;
    }
    while(num/j)
    {
        j*=10;
    }
    j/=10;
    while(num)
    {
        temp=num/j;
        num-=temp*j;
        j/=10;
        res[i]=temp+'0';
        i++;
    }
    res[i]='\0';
    printf("%s\n",res);
    return res;
}