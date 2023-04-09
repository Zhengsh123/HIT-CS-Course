import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
##保证绘图中输出中文正常
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
def generateData(mean_0,var_0,size_0,mean_1,var_1,size_1,cov=0):
    '''
    使用多维正态分布生成数据
    :param mean_0: 第0类均值，形如1*dimension
    :param var_0: 第0类方差，形如1*dimension
    :param size_0: 第0类数据及大小，标量
    :param mean_1: 第1类均值，形如1*dimension
    :param var_1:第1类方差，形如1*dimension
    :param size_1:第1类数据及大小，标量
    :param cov: 协方差
    :return: 训练特征和标签，分别形如(size0+size1)*dimension,(size0+size1)*1
    '''
    dimension=mean_0.shape[0]
    train_x=np.zeros((size_0+size_1,dimension))
    train_label=np.zeros(size_0+size_1)
    cov_0=cov*np.ones((dimension,dimension))
    cov_1 = cov * np.ones((dimension, dimension))
    for i in range(dimension):
        cov_0[i][i]=var_0[i]
        cov_1[i][i]=var_1[i]
    train_x[:size_0,:]=np.random.multivariate_normal(mean_0,cov_0,size=size_0)
    train_x[size_0:,:]=np.random.multivariate_normal(mean_1,cov_1,size=size_1)
    train_label[size_0:]=1
    train_data=np.column_stack((train_x, train_label))
    np.random.shuffle(train_data)
    return train_data[:,:-1],train_data[:,-1:]

def sigmoid(x):
    size=x.shape[0]
    res=[]
    for i in range(size):
        if x[i]>=0:
            res.append(1/(1+np.exp(-x[i])))
        else:
            res.append(np.exp(x[i])/(1+np.exp(x[i])))
    return np.array(res).reshape([size,-1])

def loss(train_x,train_y,w,lamda):
    '''
    计算需要优化的式子，归一化
    :param train_x: 训练集特征集合，形如(size0+size1)*(dimension+1)，第一列全1
    :param train_y:训练集标签，形如(size0+size1)*1
    :param w:参数 形如(dimension+1)*1
    :param lamda:正则项大小
    :return:
    '''
    size=train_x.shape[0]
    w_x=np.zeros((size,1))
    ln_x=0.0
    for i in range(size):
        w_x[i]=np.matmul(train_x[i],w)
        ln_x+=np.log(1+np.exp(w_x[i]))
    loss=np.matmul(train_y.T,w_x)-ln_x+lamda*np.matmul(w.T,w)
    return -loss
def gradient_optim(lamda, train_x, train_y, alpha, epsilon, train_num):
    '''
    :param lamda: 表示正则项系数，==0的时候表示无正则项
    :param train_x: 训练数据集 size*dimension
    :param train_y: 训练数据集 size*1
    :param alpha: 学习率
    :param epsilon: 精度
    :param train_num:训练最大次数
    :return: 优化后的参数取值
    '''
    size=train_x.shape[0]
    dimension=train_x.shape[1]
    ##为了加入w0，将X第一列全部置为1
    X=np.ones((size,dimension+1))
    X[:,1:]=train_x
    w=np.zeros((dimension+1,1))
    new_loss=loss(X,train_y,w,lamda)
    temp=0
    for i in range(train_num):
        old_loss=new_loss
        part_gradient=np.zeros((size,1))
        for j in range(size):
            part_gradient[j]=np.matmul(X[j],w)
        gradient=np.matmul(X.T,train_y-sigmoid(part_gradient))#梯度
        temp_w=w
        w=np.add(w,alpha*gradient-alpha*lamda*w)#迭代式
        new_loss=loss(X,train_y,w,lamda)
        temp=i
        if old_loss<new_loss:#迭代过程中步幅太大
            w=temp_w
            alpha/=2
            continue
        if old_loss-new_loss<epsilon:
            break
    print('梯度下降迭代次数：'+str(temp))
    w=w.reshape(dimension+1)
    fit = -(w / w[dimension])[0:dimension]  # 归一化得到系数
    return np.poly1d(fit[::-1]), w
def Hessian(X,w):
    '''
    计算当下的黑塞矩阵
    :param X: 训练数据集 size*(dimension+1)
    :param w: (dimension+1)*1
    :return: 黑塞矩阵 (dimension+1)*(dimension+1)
    '''
    size=X.shape[0]
    dimenson_1=X.shape[1]
    hessian=np.zeros((dimenson_1,dimenson_1))
    for i in range(size):
        w_x=np.matmul(X[i],w)
        for j in range(dimenson_1):
            for k in range(dimenson_1):
                p_1=sigmoid(w_x)
                hessian[j][k]+=X[i][j]*X[i][k]*p_1*(1-p_1)
    return hessian
def newton_optim(train_x, train_y, train_num):
    '''
    :param train_x: 训练数据集 size*dimension
    :param train_y: 训练数据集 size*1
    :param train_num:训练最大次数
    :return: 优化后的参数取值
    '''
    size = train_x.shape[0]
    dimension = train_x.shape[1]
    ##为了加入w0，将X第一列全部置为1
    X = np.ones((size, dimension + 1))
    X[:, 1:] = train_x
    w = np.ones((dimension + 1, 1))
    for i in range(train_num):
        part_gradient = np.zeros((size, 1))
        for j in range(size):
            part_gradient[j] = np.matmul(X[j], w)
        gradient = -np.matmul(X.T, train_y - sigmoid(part_gradient))
        hessian=Hessian(X,w)#输出黑塞矩阵
        w-=np.matmul(np.linalg.pinv(hessian),gradient)
    w = w.reshape(dimension + 1)
    fit = -(w / w[dimension])[0:dimension]  # 归一化得到系数
    return np.poly1d(fit[::-1]), w
def acc(train_x,train_y,w):
    '''
    计算w预测正确率
    :param train_x:测试集 size*dimension
    :param train_y:标签 size*1
    :param w:w 1*(dimension+1)
    :return:正确率
    '''
    size=train_x.shape[0]
    dimension=train_x.shape[1]
    acc = 0
    X=np.ones((size,dimension+1))
    X[:,1:]=train_x
    for i in range(size):
        label=0
        if np.matmul(w,X[i].T)>=0:
            label=1
        if label==train_y[i]:
            acc +=1
    return acc/size
def uci_read(path):
    '''
    :param path:数据集
    :return: 训练集和训练集标签
    '''
    column_names = ['Sample code number', 'Clump Thickness',
                    'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size',
                    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(path,names=column_names)
    data = data.replace(to_replace='?', value=np.nan)  # 非法字符的替代
    data = data.dropna(how='any')  # 去掉空值，any：出现空值行则删除
    x=data[column_names[1:10]].values.copy() ## 转换为array格式
    ##将标签中的2和4改为0和1
    y=data[column_names[10:]]
    y.loc[y['Class']==2]=0
    y.loc[y['Class'] == 4] = 1
    y=y.values.copy()
    #乱序输出
    train_data = np.column_stack((x, y))
    np.random.shuffle(train_data)
    x=train_data[:,:-1]
    y=train_data[:,-1:]
    x.astype(np.float64)
    y.astype(np.float64)
    ##生成训练集和测试集，比例为7:3
    total_num=x.shape[0]
    train_num=int(0.7*total_num)
    train_x=x[:train_num,:]
    train_y=y[:train_num,:]
    test_x=x[train_num:,:]
    test_y=y[train_num:,:]
    return train_x,train_y,test_x,test_y
def uci_run(path):
    '''
    uci数据集运行
    :param path: 数据集存储路径
    :return: 在命令行打印结果
    '''
    train_x,train_y,test_x,test_y=uci_read(path)
    lamda=0
    epsilon=1e-6
    train_num=10000
    alpha=0.01
    fit,w=gradient_optim(lamda,train_x,train_y,alpha,epsilon,train_num)
    uci_acc=acc(train_x,train_y,w)
    print('train_acc:'+str(uci_acc))
    print('test_acc:'+str(acc(test_x,test_y,w)))

def figure_2D(x,y,fit,title):
    '''
    画图——二维
    :param x: 特征集
    :param y: 标签集
    :param fit: 分类曲线
    :param title: 图名
    :return:
    '''
    y=y.reshape(x.shape[0])
    for i in range(x.shape[0]):
        if y[i]==0:
            s1=plt.scatter(x[i, 0], x[i, 1], s=50, lw=3, color='r')
        elif y[i]==1:
            s2=plt.scatter(x[i, 0], x[i, 1], s=50, lw=3, color='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend((s1,s2),('0','1'),loc='best')
    start=min(x[:, 0])
    end=max(x[:, 0])
    line_x=np.linspace(start,end,100)
    line_y=fit(line_x)
    plt.plot(line_x,line_y)
    plt.title(title)
    plt.show()
def run(optim_method):
    '''
    手工生成数据集运行
    :param optim_method: 运行的优化方法，gradient时为梯度下降，newton的时候为牛顿迭代法
    :return:
    '''
    mean0 = np.array([-0.5, -0.5])
    var0 = np.array([0.1, 0.1])
    size0 = 50
    mean1 = np.array([0.5, 0.5])
    var1 = np.array([0.1, 0.1])
    size1 = 50
    cov = 0
    lamda = 1e-9
    alpha = 0.1
    epsilon = 1e-6
    train_num = 10000
    train_x, train_y = generateData(mean0, var0, size0, mean1, var1, size1, cov)
    if optim_method=='gradient':
        fit, w = gradient_optim(lamda, train_x, train_y, alpha, epsilon, train_num)
    elif optim_method=='newton':
        fit, w = newton_optim(train_x, train_y, 200)
    train_acc = acc(train_x, train_y, w)
    figure_2D(train_x, train_y, fit, 'train_data,regulation:yes,acc:' + str(train_acc))
    ##测试集
    test_x, test_y = generateData(mean0, var0, 2 * size0, mean1, var1, size1, cov)
    test_acc = acc(test_x, test_y, w)
    figure_2D(test_x, test_y, fit, 'test_data,regulation:yes,acc:' + str(test_acc))
    print(fit)
if __name__=="__main__":
    run('gradient')
    run('newton')
    # uci_run('./breast-cancer-wisconsin.data')

