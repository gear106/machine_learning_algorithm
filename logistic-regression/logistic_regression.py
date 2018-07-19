# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 17:21:18 2018

@author: GEAR
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def Gradient_Ascent_test():
    '''
    函数说明：测试梯度上升法
    :return:
    '''

    def f_prime(x_old):
        return -2 * x_old + 4  # 原始函数的导数

    x_old = -1
    x_new = 0
    alpha = 0.01  # 学习速率，值更新幅度
    pression = 1e-6
    while abs(x_new - x_old) > pression:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)


def loadDataSet(path):
    '''
    函数说明：读取数据
    :param path: 数据文件路径
    :return:
    '''
    dataMat = []
    labelMat = []
    with open(path) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def showDataSet(path):
    dataMat, labelMat = loadDataSet(path)
    dataArr = np.array(dataMat)
    rows = dataArr.shape[0]  # 计算样本数据个数
    xcord1 = [];ycord1 = []  # class 1
    xcord2 = [];ycord2 = []  # class 0
    for i in range(rows):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('x');
    plt.ylabel('y')
    plt.legend(['class 1', 'class 0'])
    plt.show()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def gradAscent(features, labels, alpha=0.01, maxIter=500):
    '''
    函数说明：梯度上升算法
    :param features:  特征数据
    :param labels:    标签数据
    :param alpha:     学习速率
    :param maxCycles: 最大迭代次数
    :return:          权重数组
    '''
    featsMat = np.mat(features)
    labelMat = np.mat(labels).T
    rows, cols = featsMat.shape
    weights = np.ones((cols,1))    #权重系数矩阵
    weights_array = np.array([])   #记录权重系数迭代过程的数组
    for i in range(maxIter):
        h = sigmoid(featsMat * weights)
        error = labelMat - h
        weights = weights + alpha * featsMat.T * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxIter, cols)
    return weights.A, weights_array #将weights由mat转换为array

def stocGradAscent0(features, labels, alpha=0.01):
    '''
    函数说明：简单随机梯度上升法，一次只用一个样本点更新系数,
    这个随机梯度上升法的问题在于数据量比较小时权重系数可能没有
    完全收敛，此法算我们的改进版本为：stocGradAscent1
    :param features:    训练数据
    :param labels:      训练数据标签
    :param alpha:       学习速率
    :return:            权重数组和每次迭代权重数组的值
    '''
    features = np.array(features)
    labels = np.array(labels)
    rows, cols = np.shape(features)
    weights = np.ones((cols))    #权重系数为cols X 1的数组
    weights_array = np.array([])
    # sum(dataMatrix[i]*weights)为了求 f(x)的值，此处求出的 h 是一个具体的数值，而不是一个矩阵
    for i in range(rows):
        h = sigmoid(np.sum(features[i]*weights))
        error = labels[i] - h
        weights = weights + alpha * error * features[i]
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(rows, cols)
    return weights, weights_array


def stocGradAscent1(features, labels, maxIter=150):
    '''
    函数说明：改进的随机梯度上升法，一次只用一个样本点更新系数,每次随机用一个
    样本点数据进行训练
    :param features:    训练数据
    :param labels:      训练数据标签
    :param alpha:       学习速率
    :return:            权重数组和每次迭代权重数组的值
    '''
    featsMat = np.array(features)
    labelMat = np.array(labels)
    rows, cols = np.shape(features)
    weights = np.ones((cols))       # 权重系数矩阵
    weights_array = np.array([])    # 记录权重系数迭代过程的矩阵
    for j in range(maxIter):
        dataIndex = list(range(rows))
        for i in range(rows):
            alpha = 4 / (1.0 + i + j) + 0.01    #减小学习速率，每次减小1/(i+j)
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(featsMat[randIndex]*weights))
            error = labelMat[randIndex] - h
            weights = weights + alpha * error * featsMat[randIndex]
            weights_array = np.append(weights_array, weights)
            del(dataIndex[randIndex])   #删除数据
    weights_array = weights_array.reshape(maxIter*rows, cols)
    return  weights, weights_array

def showBestFit(weights, path):
    '''
    函数说明：显示分类结果
    :param weights: 由梯度上升法计算得到的权重系数
    :param path:    数据文件所在地址
    :return:
    '''
    dataMat, labelMat = loadDataSet(path)
    dataArr = np.array(dataMat)
    rows = dataArr.shape[0]  # 计算样本数据个数
    xcord1 = [];ycord1 = []  # class 1
    xcord2 = [];ycord2 = []  # class 0
    for i in range(rows):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i, 1]);ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='blue', alpha=.5)
    plt.legend(['class 1', 'class 0'])
    # 绘制决策线：
    x = np.arange(-4, 4, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def showWeight(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    path = 'D:/python/machine_learning_algorithm/logistic-regression/DataSet/testSet.txt'
    dataMat, labelMat = loadDataSet(path)
    weights1, weights_array1 = gradAscent(dataMat, labelMat)
    showBestFit(weights1, path)  # 梯度上升法
    weights2, weights_array2 = stocGradAscent1(dataMat, labelMat, maxIter=200)
    showBestFit(weights2, path)  # 改进的随机梯度上升法
    showWeight(weights_array1, weights_array2)
    weights3, weights_array3 = stocGradAscent0(dataMat, labelMat)
    showBestFit(weights3, path)  # 简单随机梯度上升法
    showWeight(weights_array2, weights_array3)