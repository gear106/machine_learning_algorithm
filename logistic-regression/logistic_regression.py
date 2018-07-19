# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 03:23:37 2018

@author: GEAR
"""
import numpy as np
import matplotlib.pyplot as plt


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

def gradAscent(features, labels):
    '''

    :param features:
    :param labels:
    :return:
    '''
    featsMat = np.mat(features)
    labelMat = np.mat(labels).T
    rows, cols = featsMat.shape
    alpha = 0.001   #学习速率
    maxCycles = 500
    weights = np.ones((cols,1))    #权重系数矩阵
    for i in range(maxCycles):
        h = sigmoid(featsMat * weights)
        error = labelMat - h
        weights = weights + alpha * featsMat.T * error

    return weights.A  #将weights由mat转换为array

def showBestFit(weights, path):
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



if __name__ == '__main__':
    path = 'D:/python/machine_learning_algorithm/logistic-regression/DataSet/testSet.txt'
    dataMat, labelMat = loadDataSet(path)
    weights = gradAscent(dataMat, labelMat)
    showBestFit(weights, path)
