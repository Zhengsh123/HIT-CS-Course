#include <stdio.h>
#include<stdlib.h>
void fib_loop_int();
void fib_loop_long();
void fib_loop_unsigned();
void fib_loop_unsigned_long();
int main()
{
    //fib_loop_int();
    // fib_loop_long();
     fib_loop_unsigned();
    // fib_loop_unsigned_long();
    system("pause");
	return 0;
}
void fib_loop_int()
{
    int a = 0, b = 1;
    for(int i = 2;i <= 100;i++)
    {
        int t = b;
       	b = a + b;
        a = t;
		printf("fib_loop(%d) = %d\n", i, b);
    }
}
void fib_loop_long()
{
    long a = 0, b = 1;
    for(int i = 2;i <= 100;i++)
    {
        long t = b;
       	b = a + b;
        a = t;
		printf("fib_loop(%ld) = %ld\n", i, b);
    }
}
void fib_loop_unsigned()
{
    unsigned a = 0, b = 1;
    for(int i = 2;i <= 100;i++)
    {
        unsigned t = b;
       	b = a + b;
        a = t;
		printf("fib_loop(%u) = %u\n", i, b);
    }
}
void fib_loop_unsigned_long()
{
    unsigned long  a = 0, b = 1;
    for(int i = 2;i <= 100;i++)
    {
        unsigned long t = b;
       	b = a + b;
        a = t;
		printf("fib_loop(%lu) = %lu\n", i, b);
    }
}