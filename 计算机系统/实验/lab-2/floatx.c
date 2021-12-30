#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include <math.h>
typedef unsigned char* byte_pointer;
void show_bytes(byte_pointer start, size_t len) 
{
	size_t i;
	for (i = 0; i < len; i++) 
    {
		printf("%.2x", start[i]);
	}
	printf("\n");
}
void show_float(float x)
{
	show_bytes((byte_pointer)&x, sizeof(float));
}
int main() 
{
	float a = +0.0f;
	float b = -0.0f;
	float c = FLT_MIN;
	float d = FLT_MAX;
	float e = FLT_MIN;
	float f = INFINITY;
	float g = NAN;
    printf("10�������Ϊ��\n");
	printf("+0:%E\n", a);
	printf("-0:%E\n", b);
	printf("��С��������:%E\n", c);
	printf("��󸡵�����:%E\n", d);
	printf("��С�������:%E\n", e);
	printf("�������:%fE\n", f);
	printf("Nan:%f.64\n", g);
    printf("\n");
	printf("16�������Ϊ��\n");
	printf("+0:");
	show_float(a);
	printf("-0:");
	show_float(b);
	printf("��С��������:");
	show_float(c);
	printf("��󸡵�����:");
	show_float(d);
	printf("��С�������:");
	show_float(e);
	printf("�������:");
	show_float(f);
	printf("Nan:");
	show_float(g);
    system("pause");
    return 0;
}
