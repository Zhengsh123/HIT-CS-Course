#include <stdio.h>
#include<stdlib.h>
int cs_atoi(char* s);
int main()
{
    char s[100];
    scanf("%s",&s);
    printf("%d\n",cs_atoi(s));
    system("pause");
    return 0;
}
int cs_atoi(char* s)
{
    int flag=0;//flag=0表示输入的是一个正数
    int i=0;
    int len=0;
    int res=0;
    if(s[0]=='-')
    {
        flag=1;
        i++;
    }
    while(s[i]!='\0')
    {
        res*=10;
        res+=(s[i]-'0');
        i++;
    }
    if(flag)
    res=-res;
    return res;
}