# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:38:42 2018

@author: GEAR
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def loadDataSet(path):
    '''
    函数说明：读取数据集
    :param path: 数据集文件所在路径
    :return:     数据集矩阵
    '''
    # 逐行读取数据集
    with open(path) as data:
        dataSet = data.readlines()
    # 计算数据集的行数和列数
    num_rows = len(dataSet)
    num_cols = len(dataSet[0].split())

    dataMat = np.zeros((num_rows, num_cols))
    index = 0
    for line in dataSet:
        # 删除默认空白符并将字符串分开
        line = line.strip().split('\t')
        dataMat[index, :] = line[:]
        index += 1
    # 这里读取的数据最好进行一下排序，方便后处理：
    index = dataMat[:, 1].argsort(0)
    dataMat = dataMat[index]
    return dataMat


def showDataSet(data):
    '''
    函数说明：显示数据集
    :param path: 数据集路径
    :return: none
    '''
    features = data[:, 1]  # 特征列
    labels = data[:, 2]  # 标签列
    plt.figure(1)
    plt.scatter(features, labels, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('features')
    plt.ylabel('labels')
    plt.show()


def standRegression(xMat, yMat):
    '''
    函数说明：根据平方误差公式计算回归方程的系数
    :param xVect:
    :param yVect:
    :return:
    '''
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('矩阵为奇异值，无法求逆')
        return
    omega = xTx.I * xMat.T * yMat
    return omega


def showRegression(xMat, yMat):
    ws = standRegression(xMat, yMat)
    yHat = (xMat * ws).A
    plt.figure(2)
    plt.plot(xMat[:, 1], yHat, c='red')
    plt.scatter(xMat[:, 1].A, yMat.A, s=20, c='blue', alpha=.5)
    plt.title('Regression')
    plt.xlabel('xValue')
    plt.ylabel('yValue')
    plt.show()


def cacuCorrcoef(xMat, yMat):
    ws = standRegression(xMat, yMat)
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat.T))  # 计算相关系数(注意这里两个向量必须为行向量？）


def lwlr(testpoint, xMat, yMat, k=1.0):
    rows = np.shape(xMat)[0]
    weights = np.mat(np.eye(rows))  # 建立权重单位阵
    for i in range(rows):  # 计算数据集汇总每个点的权重
        diffMat = testpoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * weights * xMat
    if np.linalg.det(xTx) == 0.0:
        print('矩阵为奇异矩阵，无法求逆')
        return
    ws = xTx.I * (xMat.T * weights * yMat)
    return testpoint * ws


def lwlrTest(testMat, xMat, yMat, k=1.0):
    '''
    函数说明：计算预测值
    :param testMat: 测试集
    :param xMat:    训练集
    :param yMat:    训练集
    :param k:
    :return:
    '''
    rows = np.shape(testMat)[0]
    yHat = np.zeros((rows, 1))
    for i in range(rows):
        yHat[i] = lwlr(testMat[i], xMat, yMat, k)
    return yHat


def showlwlrRegression(xMat, yMat):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    yHat_1 = lwlrTest(xMat, xMat, yMat, 0.03)
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.plot(xMat[:, 1], yHat_1, c='red')
    ax.scatter(xMat[:, 1].A, yMat.A, s=20, c='blue', alpha=.5)
    plt.xlabel('xValue')
    plt.ylabel('yValue')
    plt.title(u'lwlrRegression回归', FontProperties=font)


def lossError(yMat, yHat):
    totalLoss = ((yMat.A - yHat) ** 2).sum()
    print('totalLoss:', totalLoss)
    return totalLoss

if __name__ == '__main__':
    # 数据集所在地址：
    path = 'D:/python/machine_learning_algorithm/Regression/DateSet/abalone.txt'
    data = loadDataSet(path)
    xMat = np.mat(data[:, 0:-1])  # 特征列
    yMat = np.mat(data[:, -1]).T  # 标签列，注意这里array转matrix时列向量会变行向量，一定要注意
    yHat = lwlrTest(xMat[200:400], xMat[0:150], yMat[0:150], 10)
    lossError(yMat[200:400], yHat)

    ws = standRegression(xMat[0:200], yMat[0:200])
    yHat = (xMat[200:400] * ws).A
    plt.plot(yHat)
    plt.plot(yMat[200:400])
    plt.legend(['yhat', 'ymat'])

