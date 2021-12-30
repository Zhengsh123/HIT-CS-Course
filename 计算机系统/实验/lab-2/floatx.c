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
    printf("10进制输出为：\n");
	printf("+0:%E\n", a);
	printf("-0:%E\n", b);
	printf("最小浮点正数:%E\n", c);
	printf("最大浮点正数:%E\n", d);
	printf("最小正规格化数:%E\n", e);
	printf("正无穷大:%fE\n", f);
	printf("Nan:%f.64\n", g);
    printf("\n");
	printf("16进制输出为：\n");
	printf("+0:");
	show_float(a);
	printf("-0:");
	show_float(b);
	printf("最小浮点正数:");
	show_float(c);
	printf("最大浮点正数:");
	show_float(d);
	printf("最小正规格化数:");
	show_float(e);
	printf("正无穷大:");
	show_float(f);
	printf("Nan:");
	show_float(g);
    system("pause");
    return 0;
}
