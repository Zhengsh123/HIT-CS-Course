#include<stdio.h>
#include<stdlib.h>
#include<math.h>
//2.59题:
int function_2_59(int x,int y)
{
    int x_len=sizeof(x);
    int y_len=sizeof(y);
    int temp_x=((unsigned int)x<<((x_len-1)*8))>>((x_len-1)*8);
    int temp_y=((unsigned int)y>>8)<<8;
    return temp_x|temp_y;
}

//2.63题:
unsigned srl(unsigned x,int k)
{
    unsigned xsra=(int)x>>k;
    int w = 8 * sizeof(int);
    unsigned z = 2 << (w - k -1);
    return (z - 1) & xsra;
}
int sra(int x,int k)
{
    int xsra=(unsigned)x>>k;
    int w = 8 * sizeof(int);
    int z=1<<(w-k-1);
    int mask=z-1;
    int right=mask&xsra;
    int left=~mask&(~(z&xsra)+z);
    return left|right;
}

//2.67题:

/*
A.在许多机器上，当移位数w大于某一个数据可表示的最长字长k时，移位为w mod k,但是c语言并没有
强制要求这一点，而题中提到的机器上并没有使用通常的处理方式，移位数应该小于字长，因此出错了
*/
//2.67.B
int int_size_is_32()
{
    int set_msb=1<<31;
    int beyond_msb=1<<31;
    beyond_msb=beyond_msb<<1;
    return set_msb && !beyond_msb;
}
//2.67.C
int int_size_is_32_C()
{
    int set_msb=1<<15<<15<<1;
    int beyond_msb=1<<15<<15<<2;
    return set_msb && !beyond_msb;
}

//2.71
/* A.我们希望抽取出来的字节是有符号的，但是如果按照题目给出的那种方法扩展是无符号的*/
//2.71.B
typedef unsigned packed_t;
int xbyte(packed_t word,int bytenum)
{
    return (word<<((3-bytenum)<<3)>>24);
}

//2.75
int signed_high_prod(int x,int y);
unsigned unsigned_high_prod(unsigned x,unsigned y)
{
    int ix=(int)x;
    int iy=(int)y;
    int signed_high=signed_high_prod(ix,iy);
    int size_of_unsigned=sizeof(unsigned)<<3;
    unsigned res=signed_high+(((ix>>(size_of_unsigned-1))&1)*iy)+((iy>>((size_of_unsigned-1))&1)*ix);
    return res;
}

//2.79
int mul3div4(int x)
{
    int is_negative=x>>31;
	x=(x<<1)+x;
	if(is_negative) x+=((1<<2)-1);
	return x>>2;
}

/*2.83:
A. Y/((2^k)-1)
B.
(a).5/7
(b).2/5
(c).19/63
*/

/*2.87:
描述            HeX             M               E               V               D
-------------------------------------------------------------------------------------
-0             8000             0              -14              -0             -0.0
-------------------------------------------------------------------------------------
最小的>2的值    4001           1025/1024        1             1025*2^-9        2.001953
-------------------------------------------------------------------------------------
512            6000           1.0              9             512              512.0
-------------------------------------------------------------------------------------
最大的非规格化数 03ff          1023/1024       -14             1023*2^-24     0.000006
-------------------------------------------------------------------------------------
-无穷           fc00
-------------------------------------------------------------------------------------
十六进制表示    
为3BB0的数      3BB0           123/64           -1             123*2^-7         0.9609375
-------------------------------------------------------------------------------------
*/

/*2.91:
A.11.0010010000111111011011
B.11.001001(001循环)
C.第9位开始
*/
int main()
{
    system("pause");
    return 0;
}