#include <stdio.h>
#include<stdlib.h>
#include<math.h>
char* cs_ftoa(float num);
int main()
{
    float num;
    printf("������һ��ʮ��������");
    scanf("%f",&num);
    cs_ftoa(num);
    system("pause");
    return 0;
}
char* cs_ftoa(float num)
{
    char res[100];
    long int num1=num;//��������
    float num2=num-num1;//С������
    int temp=0;
    int i=0;
    int j=1;
    //��������ת��
    if(num<0)
    {
        res[0]='-';
        i++;
    }
    while(num1/j)
    {
        j*=10;
    }
    j/=10;
    while(num1)
    {
        temp=num1/j;
        num1-=temp*j;
        j/=10;
        res[i]=temp+'0';
        i++;
    }
    //С������ת��
    if(num2>0.001)
    {
        res[i]='.';
        i++;
        j=1;
        float num2_temp=num2;
        while(1)
        {
            num2_temp*=10;
            j++;
            if(num2_temp-(int)num2_temp<0.001)
            {
                break;
            }
        }
        for(int m=0;m<j;m++)
        {
            num2*=10;
            temp=(int)num2;
            res[i]=temp+'0';
            i++;
            num2-=temp;
        }   
    }
    res[i]='\0';
    printf("ת��Ϊ�ַ���Ϊ��%s\n",res);
    return res;
}