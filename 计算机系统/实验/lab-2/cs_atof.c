#include <stdio.h>
#include<stdlib.h>
#include <math.h>
float cs_atof(char* s);
int main()
{
    char s[100];
    scanf("%s",&s);
    printf("%f\n",cs_atof(s));
    system("pause");
    return 0;
}
float cs_atof(char* s)
{
    int flag=0;//flag=0表示输入的是一个正数
    int dec_flag=0;//这个flag=1表示进入小数位
    int i=0;
    float res=0.0;
    int dec_len=0;
    if(s[0]=='-')
    {
        flag=1;
        i++;
    }
    while(s[i]!='\0')
    {
        if(s[i]=='.')
        {
            dec_flag=1;
            i++;
        }
        res*=10;
        res+=(s[i]-'0');
        i++;
        if(dec_flag)
        {
            dec_len++;
        }
    }
    if(flag)
    res=-res;
    int temp=pow(10.0,dec_len);
    res=res/temp;
    return res;
}