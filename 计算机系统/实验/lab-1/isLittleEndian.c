#include <stdio.h>
#include<stdlib.h>
#include<stdbool.h>
bool isLittleEndian()
{
    int i = 1;
    char *a = (char *)&i;
    if(*a == 1)
        return true;
    else
        return false;
    return 0;

}
int main()
{
    if(isLittleEndian())
    {
        printf("С��\n");
    }
    else
    {
        printf("���\n");
    }
    system("pause");
    return 0;
}
