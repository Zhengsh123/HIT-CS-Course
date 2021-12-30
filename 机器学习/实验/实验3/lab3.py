import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
import scipy.stats
def generate_data(k,n,d,mu_list,sigma_list):
    """
    生成训练数据集
    :param k: k个高斯分布
    :param n: 每个高斯分布n个样本
    :param d:每个样本的特征维度
    :param mu_list:每个高斯分布的均值的list，格式为长度为k的向量，每个元素都是d维
    :param sigma_list:每个高斯分布的方差的list，格式为长度为k的向量，每个元素都是d*d维
    :return:带标签的训练集，形如[n*k,d]
             标签集合，形如[n*k,1]
    """
    X=np.zeros((k*n,d+1))
    for i in range(k):
        X[i*n:(i+1)*n,:d]=np.random.multivariate_normal(mu_list[i], sigma_list[i], size=n)
        X[i * n:(i+1)*n, d:d+1] = i
    #乱序输出训练数据集
    np.random.shuffle(X)
    return X[:,:d],X[:,d:d+1]
def kmeans(X,k,epsilon=1e-5):
    """
    k-means算法实现，计算分类结果和中心
    :param X: 训练样本，[n,d]
    :param k:预计要分成多少个簇
    :param epsilon:精度
    :return:预测的标签label，格式为[n*1],与训练样本一一对应
           center：每一类的中心，格式为[k*d]
    """
    n=X.shape[0]
    dimension=X.shape[1]
    center=np.zeros((k,dimension))
    label=np.zeros((n,1))
    #随机初始化center,从X中选取k个作为中心
    random_index=np.random.choice(range(0,n),k,replace=False)
    for i in range(k):
        center[i,:]=X[random_index[i],:]
    iter=0
    while True:
        iter+=1
        distance=np.zeros(k)#标记每个X中的样本和k个中心点距离
        #重新计算每个样本的标签
        for i in range(n):
            for j in range(k):
                distance[j]=np.linalg.norm(X[i,:]-center[j,:])#默认使用二范数
            label[i,0]=np.argmin(distance)
        label=label.astype(np.int16)
        #计算样本的中心点
        new_center=np.zeros((k,dimension))
        num=np.zeros(k)#记录每个簇现在有多少个元素
        for i in range(n):
            new_center[label[i,0],:]=new_center[label[i,0],:]+X[i,:]
            num[label[i,0]]+=1
        for i in range(k):
            new_center[i,:]=new_center[i,:]/num[i]
        if np.linalg.norm(new_center - center,ord=2) < epsilon:  # 二范数计算精度，一定范围内表示不再变化
            break
        else:
            center = new_center
    print(iter)
    return label,center
def E(X,pi_list,mu_list,sigma_list):
    """
    EM算法中的E步,计算样本由每个高斯分布生成的后验概率
    :param X:训练数据集，格式为n*d
    :param pi_list:混合系数矩阵，格式为长度为k的向量
    :param mu_list:每个高斯分布的均值的list，格式为长度为k的向量，每个元素都是d维
    :param sigma_list:每个高斯分布的方差的list，格式为长度为k的向量，每个元素都是d*d维
    :return: 样本由各个混合高斯成分生成的后验概率，格式为n*k
    """
    n=X.shape[0]
    d=X.shape[1]
    k=pi_list.shape[0]
    gamma_z=np.zeros((n,k))
    for i in range(n):
        pdf_sum=0
        pdf=np.zeros(k)
        for j in range(k):
            pdf[j]=scipy.stats.multivariate_normal.pdf(X[i],mean=mu_list[j],cov=sigma_list[j])#计算pdf
            pdf_sum+=pdf[j]
        for j in range(k):
            gamma_z[i,j]=pdf[j]/pdf_sum
    return gamma_z
def M(X, gamma_z, mu_list):
    """
    EM算法中的M步
    :param X:训练集，格式为n*d
    :param gamma_z:后验概率，格式为n*k
    :param mu_list:均值list，格式为长度为k的向量，每个元素都是d维
    :return:mu_list，sigmma_list，pi_list的新估计
    """
    n=X.shape[0]
    d=X.shape[1]
    k=gamma_z.shape[1]
    new_mu_list=np.zeros((k,d))
    new_sigmma_list=np.zeros((k,d,d))
    new_pi_list=np.zeros(k)
    for j in range(k):
        n_j=np.sum(gamma_z[0:,j])
        new_pi_list[j]=n_j/n
        gamma=gamma_z[:,j].reshape(1,n) #1*n
        new_mu_list[j,:]=np.matmul(gamma,X)/n_j
        temp=X-mu_list[j] #n*d
        new_sigmma_list[j]=np.matmul(temp.T,np.multiply(temp,gamma.reshape(n,1)))/n_j
    return new_mu_list,new_sigmma_list,new_pi_list
def cal_likelihood(X, mu_list, sigma_list, pi_list):
    """
    计算最大似然函数（也就是GMM想要优化的内容）
    :param X: 训练集，格式为n*d
    :param mu_list: 均值，格式为k*d
    :param sigma_list: 方差，格式为 k*d*d
    :param pi_list: 混合成分，长度为k的向量
    :return: 最大似然函数的值
    """
    ll=0
    n=X.shape[0]
    k=mu_list.shape[0]
    for i in range(n):
        pdf = np.zeros(k)
        temp_sum=0
        for j in range(k):
            pdf[j] = scipy.stats.multivariate_normal.pdf(X[j], mean=mu_list[j], cov=sigma_list[j])
            temp_sum+=pi_list[j]*pdf[j]
        ll+=np.log(temp_sum)
    return ll
def GMM(X,k,max_iter=100,epsilon=1e-5):
    """
    GMM算法实现
    :param X: 训练集，格式为n*d
    :param k: 预计有多少个簇
    :param max_iter: 最大迭代次数
    :param epsilon: 精度
    :return: 预测的标签label，格式为[n*1],与训练样本一一对应
             center：每一类的中心，格式为[k*d]
    """
    ##初始化
    n=X.shape[0]
    pi_list=np.ones(k)/k
    sigma_list=np.array([0.1 * np.eye(X.shape[1])] * k)
    ##mu_list初始化用kmeans来做
    label,mu_list=kmeans(X,k,epsilon)
    old_ll=cal_likelihood(X, mu_list, sigma_list, pi_list)
    gamma_z=np.zeros((n,k))
    iter=0
    for i in range(max_iter):
        iter+=1
        gamma_z=E(X,pi_list,mu_list,sigma_list)
        mu_list,sigma_list,pi_list=M(X,gamma_z,mu_list)

        new_ll=cal_likelihood(X, mu_list, sigma_list, pi_list)
        if old_ll < new_ll and new_ll - old_ll < epsilon:
            break
        old_ll=new_ll
    for i in range(n):
        label[i]=np.argmax(gamma_z[i,:])
    print(iter)
    return label,mu_list

def acc(real_label,pred_label,k):
    """
    计算聚类精度
    :param real_label:真实标签，n*1
    :param pred_label: 预测标签，n*1
    :param k: 预测标签中分了多少类
    :return: 聚类精度
    """
    ##计算聚类精度的时候采用Purity计算
    ##需要注意的是找所有可能中对应的正确率最大的那一个
    assert real_label.shape[0]==pred_label.shape[0]
    n=real_label.shape[0]
    permu=list(permutations(range(k), k))#生成预测标签的全排列
    count=np.zeros(len(permu))
    for i in range(len(permu)):
        for j in range(n):
            if real_label[j]==permu[i][pred_label[j][0]]:
                count[i]+=1
    return np.max(count)/n
def uci_read(path):
    '''
    :param path:数据集
    :return: 训练集和训练集标签
    '''
    column_names = ['sepal length in cm', 'sepal width in cm',
                    ' petal length in cm', 'petal width in cm',
                    'Class']
    data = pd.read_csv(path,names=column_names)
    data = data.replace(to_replace='?', value=np.nan)  # 非法字符的替代
    data = data.dropna(how='any')  # 去掉空值，any：出现空值行则删除
    x=data[column_names[0:4]].values.copy() ## 转换为array格式
    ##将分类信息变为数字信息
    y=data[column_names[4:]]
    y.loc[y['Class']=='Iris-setosa']=0
    y.loc[y['Class'] == 'Iris-versicolor'] = 1
    y.loc[y['Class'] == 'Iris-virginica'] = 2
    y=y.values.copy()
    #乱序输出
    train_data = np.column_stack((x, y))
    np.random.shuffle(train_data)
    x=train_data[:,:-1]
    y=train_data[:,-1:]
    x.astype(np.float64)
    y.astype(np.float64)
    return x,y
def uci_run(path,epsilon=1e-5):
    X,label=uci_read(path)
    acc_kmeans_list=[]
    acc_GMM_list=[]
    for i in range(1,10):
        pred_label_kmeans,center_kmeans = kmeans(X,i,epsilon)
        pred_label_GMM,center_GMM = GMM(X,i,epsilon=epsilon)
        acc_kmeans_i=acc(label,pred_label_kmeans,i)
        acc_GMM_i=acc(label,pred_label_GMM,i)
        acc_kmeans_list.append(acc_kmeans_i)
        acc_GMM_list.append(acc_GMM_i)

    kmeans_best_k=np.argmin(np.array(acc_kmeans_list))+1
    GMM_best_k = np.argmin(np.array(acc_GMM_list)) + 1

    pred_label_kmeans, center_kmeans = kmeans(X, kmeans_best_k, epsilon)
    pred_label_GMM, center_GMM = GMM(X, GMM_best_k, epsilon=epsilon)

    title_kmeans_i="kmeans,acc="+str(acc_kmeans_list[kmeans_best_k])+",k="+str(kmeans_best_k)+",epsilon="+str(epsilon)
    title_GMM_i = "GMM,acc=" + str(acc_GMM_list[GMM_best_k]) + ",k=" + str(GMM_best_k) + ",epsilon=" + str(epsilon)
    show(X,pred_label_kmeans,center_kmeans,title_kmeans_i)
    show(X, pred_label_GMM, center_GMM, title_GMM_i)

def show(X,label,center=None,title=None):
    """
    画图函数
    :param X:数据集，本实验中格式为[k*n,d]
    :param label:标签集，格式为[k*n,1]，和数据集中数据一一对应
    :param center: 每个簇的中心点坐标，本实验中格式为[k,d]
    :param title:图名
    :return:
    """
    plt.scatter(X[:,0], X[:,1], c=label[:,0], marker='.', s=25)
    if not center is None:
        plt.scatter(center[:, 0], center[:, 1], c='r', marker='x', s=250)
    if not title is None:
        plt.title(title)
    plt.show()

if __name__=="__main__":
    k=3
    n=200
    d=2
    mu_list=[ [1,3], [2,5], [5,2] ]
    sigma_list=[ [[1,0],[0,2]], [[2,0],[0,3]], [[3,0],[0,4]] ]
    X,label=generate_data(k,n,d,mu_list,sigma_list)

    # pred_label,center=kmeans(X,3)
    # print(acc(label,pred_label,3))
    # show(X, label,center)
    # pred_label,center=GMM(X,k)
    # print(acc(label, pred_label, 3))
    # show(X, label, center)
    path='./iris.data'
    uci_run(path)