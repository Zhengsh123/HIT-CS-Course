#include<stdio.h>

#define BUF_SIZE 64
/*3.59
首先，我们可以把x写成x=2^64*xh+xl,y=2^64*yh+yl,那么x*y可以在128位精度下可以写成
x*y=2^64*(xh*yl+xl*yh)+xl*yl,这一段汇编代码就是基于这个式子得到的
store_prod:
    movq   %rdx, %rax   # %rax = y
    cqto                # %rax符号扩展,%rdx = yh,%rax = yl
    movq   %rsi, %rcx   # %rcx = x
    sarq   $63,  %rcx   # 将%rcx算术右移63位,做了符号位扩展,%rcx = xh,%rsi = xl
    imulq  %rax, %rcx   # %rcx = yl * xh
    imulq  %rsi, %rdx   # %rdx = xl * yh
    addq   %rdx, %rcx   # %rcx = yl * xh + xl * yh
    mulq   %rsi         # 无符号计算 xl*yl,将xl*yl的128位结果的高位放在%rdx,低位放在%rax
    addq   %rcx, %rdx   # 将xh*yl+xl*yh加到%rdx
    movq   %rax, (%rdi) # 将%rax的值放到dest的低位
    movq   %rdx, 8(%rdi)# 将%rdx的值放到dest的高位，则dest的值就表示最后的结果
    ret
*/

//3.61
long cread_alt(long *xp)
{
    long i=0;
    long *p=(xp ? xp: &i);
    return *p;
}

//3.63
long switch_prob(long x, long n) 
{
	long result = x;
	switch (n)  
    {
	case 60:
	case 62:
		result = x * 8;
		break;
	case 63:
		result = result >> 3;
		break;
	case 64:
		result = (result << 4) - x;
		x = result;
	case 65:
		x = x * x;
	default:
		result = x + 75;
	}
    return result;
}

/*3.65
A.%rdx
B.%rax
C.15
*/

/*3.67
A. %rsp = x;
   %rsp + 8 = y;
   %rsp + 16 = &z;
   %rsp + 24 = z;
B.%rsp + 64;
C.使用%rsp+地址偏置量
D.%rdi+地址偏置量
E.%rsp + 64 = y
   %rsp + 72 = x
   %rsp + 80 =z
F.均使用结构体首地址作为给函数传递的参数和返回的参数
*/

/*3.69
A.
CNT=7
B.
typedef struct {
	long idx;
	long x[4];
}a_struct;
*/

//3.71
void good_echo() 
{
	char buf[BUF_SIZE];
	char* p = fgets(buf, BUF_SIZE, stdin);
	if (ferror(stdin) || p == NULL)
	{
		printf("%s", "error");
		return;
	}
	printf("%s", p);
	return;
}
int main()
{
    good_echo();
    return 0;
}