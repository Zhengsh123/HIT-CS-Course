import numpy as np
import matplotlib.pyplot as plt
##保证绘图中输出中文正常
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
## 生成数据并加入噪声,sin(2pix),同时生成训练数据集
def generate(degree,size,begin=0,end=1,mu=0,sigma=0.2):
    '''
    :param degree:拟合的阶数
    :param size:数据集大小
    :param begin:数据集开始点
    :param end:结束点
    :param mu:噪音的均值
    :param sigma:噪音方差
    :return:无噪音数据集，加上噪音数据集
    '''
    x = np.linspace(begin, end, size) #原始数据集
    noise=np.random.normal(mu,sigma,size)#噪音
    y = np.sin(2 * np.pi * x)+noise #原始数据集
    degree_list = np.arange(0, degree + 1)  # 阶数list
    x_train = np.zeros((size, degree + 1))  # 训练数据集，size*(degree+1)
    for i in range(size):
        temp=np.ones(degree+1)*x[i]
        x_train[i]=temp**degree_list
    y_train = y.reshape(size, 1)  # 训练数据集，size*1
    return x,y,x_train,y_train
##计算loss
def loss(x_train,y_train,w,lamda):
    '''
    :param x_train: 训练数据集
    :param y_train: 训练数据集
    :param w: 估计参数
    :param lamda: 正则项系数
    :return:loss
    '''
    temp=np.matmul(x_train,w)-y_train
    return (np.matmul(temp.T,temp)+lamda*np.matmul(w.T,w))/2
##解析法计算参数
def resolve(lamda, x_train, y_train):
    '''
    :param lamda: 表示正则项系数，==0的时候表示无正则项
    :param x_train:训练数据集
    :param y_train:训练数据集
    :return:参数w,解析解的结果
    '''
    w=np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.T,x_train)+
                               lamda*np.eye(x_train.shape[1])),x_train.T),y_train)
    return w,np.poly1d(w[::-1].reshape(x_train.shape[1]))
##梯度下降求最优解
def gradient_optim(lamda,x_train,y_train,alpha,epsilon,train_num):
    '''
    :param lamda: 表示正则项系数，==0的时候表示无正则项
    :param x_train: 训练数据集
    :param y_train: 训练数据集
    :param alpha: 学习率
    :param epsilon: 精度
    :param train_num:训练最大次数
    :return: 优化后的参数取值
    '''
    w=np.zeros((x_train.shape[1],1)) #(degree+1)*1
    new_loss=loss(x_train,y_train,w,lamda)
    temp=0
    for i in range(train_num):
        old_loss=new_loss
        gradient=np.matmul(np.matmul(x_train.T,x_train),w)-np.matmul(x_train.T,y_train)+lamda*w
        temp_w=w
        w-=alpha*gradient
        new_loss=loss(x_train,y_train,w,lamda)
        temp=i
        if new_loss-old_loss>0:##新一步的loss比之前一步loss大，学习率太大
          w=temp_w
          alpha/=2
        if old_loss-new_loss<epsilon:##达到精度要求，停止训练
            break
    print('梯度下降代数：'+str(temp))
    return np.poly1d(w[::-1].reshape(x_train.shape[1]))

##共轭梯度法优化
def conjugate(lamda,x_train,y_train,epsilon):
    '''
    :param lamda: 表示正则项系数，==0的时候表示无正则项
    :param x_train: 训练集
    :param y_train:训练集
    :param epsilon:训练精度
    :return:优化结果
    '''
    # Aw=b 其中 A=x'x+lamda，b=x'y
    A=np.matmul(x_train.T,x_train)+lamda*np.eye(x_train.shape[1])#(degree+1)*(degree+1)
    b=np.matmul(x_train.T,y_train)#(degree+1)*1
    w=np.zeros((x_train.shape[1],1))
    r=b
    p=b
    k=0
    while True:
        k=k+1
        temp=np.matmul(r.T,r)
        a=np.matmul(r.T,r)/(np.matmul(np.matmul(p.T,A),p))
        w=w+a*p
        r=r-np.matmul(a*A,p)
        if np.matmul(r.T,r)<epsilon:
            break
        b=np.matmul(r.T,r)/temp
        p=r+b*p
    return np.poly1d(w[::-1].reshape(x_train.shape[1]))

def figure_show(x, y,fit, title):
    '''
    :param x:数据集
    :param y: 数据集
    :param fit: 估计多项式
    :param title: 图名
    :return: 绘图
    '''
    plt.plot(x, y,'co', label='noise_data')
    real_x = np.linspace(0, 1, 200)
    real_y = np.sin(real_x * 2 * np.pi)
    fit_y = fit(real_x)
    plt.plot(real_x, fit_y, 'black', label='fit_result')
    plt.plot(real_x, real_y, 'r', label='real_result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title(title)
    plt.show()

def loss_show(x_train,y_train,x_test,y_test,train_size,test_size):
    '''
    :param x_train: 训练集
    :param y_train: 训练集
    :param x_test: 测试集
    :param y_test: 测试集
    :param train_size: 训练集大小
    :param test_size: 测试集大小
    :return: 绘图
    '''
    ln_lamda=list(np.linspace(-20,-1,20))
    loss_train = []#size=20
    loss_test = []#size=20
    for i in range(20):
        lamda=np.exp(ln_lamda[i])
        w,fit=resolve(lamda, x_train, y_train)
        loss_train.append(float(loss(x_train,y_train,w,lamda)/train_size))
        loss_test.append(float(loss(x_test,y_test,w,lamda)/test_size))
    plt.plot(ln_lamda, loss_train, 'black', label='train')
    plt.plot(ln_lamda, loss_test, 'b', label='test')
    min_test_loss=min(loss_test)
    min_loss_index=loss_test.index(min_test_loss)
    lamda=min_loss_index-20
    plt.xlabel('$\ln \lambda$')
    plt.ylabel('$E_w$')
    plt.legend(loc=2)
    plt.title('$Min: e^{' + str(lamda) + '}$')
    plt.show()

def draw(x,y,fit,method,lamda,size,degree):
    '''
    :param x:数据集
    :param y:数据集
    :param fit:估计式子
    :param method:估计方法名称，str
    :param lamda:正则项系数
    :param size:数据集大小
    :param degree:估计多项式次数
    :return:
    '''
    if lamda!=0:
        lamda='yes'
    else:
        lamda = 'no'
    title=method+'  regularization:'+lamda+'  dataset_size:'+str(size)+'  degree:'+str(degree)

    figure_show(x,y,fit,title)
def compare_draw(x,y,fit1,fit2,fit3,lamda,size,degree):
    '''
    :param x:数据集
    :param y:数据集
    :param fit1:第一种估计（在本例中为解析法）
    :param fit2:第二种估计
    :param fit3:第三种估计
    :param lamda:正则项系数
    :param size:数据集规模
    :param degree:估计的次数
    :return:画图
    '''
    plt.plot(x, y, 'co', label='noise_data')
    real_x = np.linspace(0, 1, 100)
    real_y = np.sin(2 * np.pi*real_x )
    fit_y_1 = fit1(real_x)
    fit_y_2 = fit2(real_x)
    fit_y_3 = fit3(real_x)
    plt.plot(real_x, fit_y_1, 'b', label='RESOLVE_fit_result')
    plt.plot(real_x, fit_y_2, 'r', label='GD_fit_result',linestyle='--')
    plt.plot(real_x, fit_y_3, 'y', label='CG_fit_result',linestyle='--')
    plt.plot(real_x, real_y, 'g', label='real_result',linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.title('regularization:'+str(lamda)+'  dataset_size:'+str(size)+'  degree:'+str(degree))
    plt.show()
if __name__=="__main__":
    degree=3
    size=10
    alpha=0.001
    epsilon=1e-10
    train_num=100000
    lamda= 0

    x,y,x_test,y_test=generate(degree,20,begin=0,end=1)
    x, y, x_train, y_train = generate(degree, size)
    w,fit1=resolve(lamda,x_train,y_train)
    # fit2=gradient_optim(lamda,x_train,y_train,alpha,epsilon,train_num)
    # fit3=conjugate(lamda,x_train,y_train,epsilon)
    # loss_show(x_train,y_train,x_test,y_test,size,size)
    draw(x,y,fit1,'resolve',lamda,size,degree)
    # draw(x, y, fit2, 'gradient descent', lamda, size, degree)
    # draw(x, y, fit3, 'conjugate gradient', lamda, size, degree)


