%% romberg法
clear;
format long;
sym(x);
f=input('请输入被积函数f(x)=');
a=input('请输入积分下限a=');
b=input('请输入积分上限b=');
e=input('请输入精度要求e=');
t=0;
k=0;

x=a;
fa=eval(f);
x=b;
fb=eval(f);
T(1)=(b-a)*(fa+fb)/2;
while (t==0)
    temp=T(1);
    k=k+1;
    sum=0;
    N=2^k;
    h=(b-a)/N;
    for i=0:N-1
        x=a+h*(i+0.5);
        fsum=eval(f);
        sum=sum+fsum;
    end
    T(k+1)=0.5*T(k)+0.5*h*sum;
    for j=k:-1:1
        T(j)=(4^(k+1-j)*T(j+1)-T(j))/(4^(k+1-j)-1);
    end
    if abs(temp-T(1))<=e
        t=t+1;
    end
end
fprintf('数值积分结果为%.8f\n',T(1));

l=length(T);
T(1)=(b-a)*(fa+fb)/2;
for i=2:l
    sum=0;
    N=2^(i-1);
    h=(b-a)/N;
    for j=0:N-1
        x=a+h*(j+0.5);
        fsum=eval(f);
        sum=sum+fsum;
    end
    T(i)=0.5*T(i-1)+0.5*h*sum;
end
for i=1:l
    fprintf('T(0,%d)=%.8f  ',i-1,T(i));
end
fprintf('\n');
for j=2:l
    for i=1:l+1-j
        T(i)=(4^(j-1)*T(i+1)-T(i))/(4^(j-1)-1);
        fprintf('T(%d,%d)=%.8f  ',j-1,i-1,T(i));
    end
    fprintf('\n');
end
        