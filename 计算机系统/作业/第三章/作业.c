#include<stdio.h>

#define BUF_SIZE 64
/*3.59
���ȣ����ǿ��԰�xд��x=2^64*xh+xl,y=2^64*yh+yl,��ôx*y������128λ�����¿���д��
x*y=2^64*(xh*yl+xl*yh)+xl*yl,��һ�λ�������ǻ������ʽ�ӵõ���
store_prod:
    movq   %rdx, %rax   # %rax = y
    cqto                # %rax������չ,%rdx = yh,%rax = yl
    movq   %rsi, %rcx   # %rcx = x
    sarq   $63,  %rcx   # ��%rcx��������63λ,���˷���λ��չ,%rcx = xh,%rsi = xl
    imulq  %rax, %rcx   # %rcx = yl * xh
    imulq  %rsi, %rdx   # %rdx = xl * yh
    addq   %rdx, %rcx   # %rcx = yl * xh + xl * yh
    mulq   %rsi         # �޷��ż��� xl*yl,��xl*yl��128λ����ĸ�λ����%rdx,��λ����%rax
    addq   %rcx, %rdx   # ��xh*yl+xl*yh�ӵ�%rdx
    movq   %rax, (%rdi) # ��%rax��ֵ�ŵ�dest�ĵ�λ
    movq   %rdx, 8(%rdi)# ��%rdx��ֵ�ŵ�dest�ĸ�λ����dest��ֵ�ͱ�ʾ���Ľ��
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
C.ʹ��%rsp+��ַƫ����
D.%rdi+��ַƫ����
E.%rsp + 64 = y
   %rsp + 72 = x
   %rsp + 80 =z
F.��ʹ�ýṹ���׵�ַ��Ϊ���������ݵĲ����ͷ��صĲ���
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