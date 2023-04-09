import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
import math
def rotate(X,theta):
    """
    旋转数据
    :param X:数据集，格式为3*n
    :param theta:需要旋转的角度
    :return:旋转后的数据，格式为3*n
    """
    rotate = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    return np.dot(rotate,X)
def generate_data(num=100,y_scale=100):
    """
    生成瑞士卷数据
    :param num: 生成的数据点的数量
    :param y_scale:生成的瑞士卷的厚度
    :return: 数据集，格式为n*3
    """
    t=np.pi*(1+2*np.random.rand(1,num))
    x=t*np.cos(t)
    y=y_scale*np.random.rand(1,num)
    z=t*np.sin(t)
    X=np.concatenate((x,y,z))
    X=rotate(X,60)
    X=X.T
    return X
def generate_guass_data(num):
    """
    生成三维高斯分布数据，并旋转
    :param num:需要的数据点个数
    :return:数据集，格式为n*3
    """
    mean=[5,4,-3]
    cov=[[0.1,0,0],[0,1,0],[0,0,1]]
    X=np.random.multivariate_normal(mean, cov, size=num).T#3*num
    X=rotate(X,40 * np.pi / 180).T
    return X
def show_3D(X):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=20, azim=80)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0])
    ax.legend(loc='best')
    plt.show()
def show_3D_change_view(X):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=30, azim=140)
    ax.scatter(X[:,0], X[:,1], X[:,2], c=X[:,0])
    ax.legend(loc='best')
    plt.show()
def PCA(X,k):
    """
    PCA降维
    :param X:数据，格式为n*d,n表示数据集中数据个数，d表示维度
    :param k: 需要降到的维度数量，满足k<=d
    :return:降维后的结果
    """
    n=X.shape[0]
    d=X.shape[1]
    assert d>=k
    #零均值化
    X_mean=np.sum(X,0)/n
    X=X-X_mean
    ##求协方差矩阵
    Cov=np.dot(X.T,X)#d*d
    ##求特征值和特征向量
    eigValues,featureVectors=np.linalg.eig(Cov)
    ##特征值排序
    eigSort=np.argsort(eigValues)
    ##选前k个特征值对应的特征向量
    eigRes=featureVectors[:,eigSort[:-(k+1):-1]]#d*k
    ##计算PCA结果
    pca_res=np.dot(X,eigRes)##n*k
    return X,eigRes,X_mean
def show(X):
    plt.scatter(X[:, 0].tolist(), X[:, 1].tolist(), c=X[:, 0].tolist())
    plt.show()
def self_data_run():
    X=generate_data(2000)
    X,eigRes,X_mean=PCA(X,2)
    show(np.dot(X,eigRes))
def read_data(path,compress_size):
    """
    读取人脸数据集
    :param path: 人脸数据集地址
    :param compress_size: 图片压缩比例，格式为(,)
    :return: 训练数据集
    """
    file_list=os.listdir(path)
    X_list=[]
    i=1
    for file in file_list:
        plt.subplot(3,3,i)
        file_path=os.path.join(path,file)
        img=cv2.resize(cv2.imread(file_path),compress_size)
        img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.imshow(img_grey,cmap="gray")
        img_temp=img_grey.reshape(-1)#展开为一维数据
        X_list.append(img_temp)
        i+=1
    plt.show()
    return np.asarray(X_list)

def face_data_run():
    compress_size=(50,50)
    X=read_data('./data',compress_size)
    n,d=X.shape
    X,eigRes,X_mean=PCA(X,2)
    #降维
    eigRes=np.real(eigRes)
    pca=np.dot(X,eigRes)##n*k
    ##重构
    recons=np.dot(pca,eigRes.T)+X_mean
    for i in range(n):
        plt.subplot(3,3,i+1)
        plt.imshow(recons[i].reshape(compress_size),cmap="gray")
    plt.show()
def PSNR(X_real,X_recon):
    """
    计算信噪比
    :param X_real: 未降维的图片
    :param X_recon: 重构图片
    :return: 信噪比值
    """
    mse=np.mean((X_real/255.-X_recon/255.)**2)
    MAX=1
    return 20*math.log10(MAX/math.sqrt(mse))
if __name__=="__main__":
    read_data('./data',(50,50))