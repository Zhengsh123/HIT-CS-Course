#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int ID_int=1190300321;
float ID_float=1190300321119;
double ID_double=1190300321119;
unsigned ID_unsigned=1190300321;
short ID_short=1190300321;
char name_char='s';
char Name_string[]="郑晟赫";
int* I_ID_pointer=&ID_int;
typedef unsigned char *byte_pointer;
struct Student
{
   int ID_int;
   char Name_char[];
} Student_Struct;
union Student1{
   int ID_int;
}Student_union;
enum week_enum{ Mon, Tues, Wed, Thurs, Fri, Sat, Sun }Week_enum;
void show_bytes(byte_pointer start,size_t len)
{
    size_t i;
    for(i=0;i<len;i++)
    {
        printf("%.2x",start[i]);
    }
    printf("\t");
}
int main()
{
    Student_Struct.ID_int=1190300321;
    Student_union.ID_int=1190300321;
    Week_enum=Thurs;
    printf("%s\t                %s\t                %s\t                %s\t","变量名","变量值","地址","16进制");
    printf("\n");
    //int
    printf("%s\t                %d\t        %x\t                ","ID_int",ID_int,&ID_int);
    show_bytes((byte_pointer)&ID_int,sizeof(ID_int));
    //unsigned
    printf("\n");
    printf("%s\t        %d\t        %x\t                ","ID_unsigned",ID_unsigned,&ID_unsigned);
    show_bytes((byte_pointer)&ID_unsigned,sizeof(ID_unsigned));
    //short
    printf("\n");
    printf("%s\t        %hd\t                %x\t                ","ID_short",ID_short,&ID_short);
    show_bytes((byte_pointer)&ID_short,sizeof(ID_short));
    //float
    printf("\n");
    printf("%s\t        %.2f\t%x\t                ","ID_float",ID_float,&ID_float);
    show_bytes((byte_pointer)&ID_float,sizeof(ID_float));
    //double
    printf("\n");
    printf("%s\t        %.2lf\t%x\t                ","ID_double",ID_double,&ID_double);
    show_bytes((byte_pointer)&ID_double,sizeof(ID_double));
    //char
    printf("\n");
    printf("%s\t        %c\t                %x\t                ","name_char",name_char,&name_char);
    show_bytes((byte_pointer)&name_char,sizeof(name_char));
    //string
    printf("\n");
    printf("%s\t        %s\t                %x\t                ","Name_string",Name_string,&Name_string);
    show_bytes((byte_pointer)&Name_string,sizeof(Name_string));
    //pointer
    printf("\n");
    printf("%s\t        %x\t                %x\t                ","I_ID_pointer",I_ID_pointer,&I_ID_pointer);
    show_bytes((byte_pointer)&I_ID_pointer,sizeof(I_ID_pointer));
    //struct
    printf("\n");
    printf("%s\t        %x\t        %x\t                ","Student_Struct",Student_Struct,&Student_Struct);
    show_bytes((byte_pointer)&Student_Struct,sizeof(Student_Struct));
    //union
    printf("\n");
    printf("%s\t        %x\t        %x\t                ","Student_union",Student_union,&Student_union);
    show_bytes((byte_pointer)&Student_union,sizeof(Student_union));
    //enum
    printf("\n");
    printf("%s\t        %x\t                %x\t                ","Week_enum",Week_enum,&Week_enum);
    show_bytes((byte_pointer)&Week_enum,sizeof(Week_enum));
    //main
    printf("\n");
    printf("%s\t                %x\t                %x\t                ","main",main,&main);
    printf("%x",&main);
    //printf
    printf("\n");
    printf("%s\t                %x\t                %x\t                ","printf",printf,&printf);
    printf("%x",&printf);
    printf("\n");
    system("pause");
    return 0;
}
