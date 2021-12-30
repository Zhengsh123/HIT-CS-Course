#include <stdio.h>
#include<stdlib.h>
int utf8len(char* cstr) {
	char *s = cstr;
	int length = 0;
	char num;
	while (*s != '\0') 
    {
		length++;
		num = *s;
		if (!(num & 0x80)) 
        {
			s++;
		}
		else 
        {
            int result = 1;
            num=num<<1;
            while(num&0x80)
            {
                num = num << 1;
                result++;
            }   
			s += result;
		}
	}
	return length;
}
int main() {
	char cstr[100];
	scanf("%s", cstr);
	printf("%d\n", utf8len(cstr));
    system("pause");
	return 0;
}
