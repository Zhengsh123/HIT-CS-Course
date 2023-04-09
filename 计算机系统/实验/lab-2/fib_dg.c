#include <stdio.h>
#include<stdlib.h>
int fib_dg_int(int x);
long fib_dg_long(long x);
unsigned int fib_dg_unsigned_int(unsigned int x);
unsigned long fib_dg_unsigned_long(unsigned long x);
int main()
{
    // for(int i=0;i<100;i++)
    // {
    //     printf("%d,%d\n",fib_dg_int(i),i);
    // }
    // for(unsigned int i=0;i<100;i++)
    // {
    //     printf("%u,%u\n",fib_dg_unsigned_int(i),i);
    // }
    // for(long i=0;i<100;i++)
    // {
    //     printf("%ld,%ld\n",fib_dg_long(i),i);
    // }
    for(unsigned long i=0;i<100;i++)
    {
        printf("%lu,%lu\n",fib_dg_int(i),i);
    }
    system("pause");
    return 0;
}
//intÐÍ
int fib_dg_int(int x)
{
    if(x <= 0)
    {
        return 0;
    }
	if(x == 1)
    {
		return 1;
	}
	return fib_dg_int(x - 1) + fib_dg_int(x - 2);
}
//longÐÍ
long fib_dg_long(long x)
{
    if(x <= 0)
    {
        return 0;
    }
	if(x == 1)
    {
		return 1;
	}
	return fib_dg_long(x - 1) + fib_dg_long(x - 2);
}
//unsigned int
unsigned int fib_dg_unsigned_int(unsigned int x)
{
    if(x <= 0)
    {
        return 0;
    }
	if(x == 1)
    {
		return 1;
	}
	return fib_dg_unsigned_int(x - 1) + fib_dg_unsigned_int(x - 2);
}
//unsigned long
unsigned long fib_dg_unsigned_long(unsigned long x)
{
    if(x <= 0)
    {
        return 0;
    }
	if(x == 1)
    {
		return 1;
	}
	return fib_dg_unsigned_long(x - 1) + fib_dg_unsigned_long(x - 2);
}