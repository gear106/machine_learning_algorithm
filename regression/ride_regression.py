# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:42:50 2018

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
    return dataMat

def regularize(xMat, yMat):
    '''
    函数说明：数据归一化
    :param xMat: 输入特征数据
    :param yMat: 输入结果数据
    :return:     归一化后的数据
    '''
    xMean = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)  # 按列计算方差
    rxMat = (xMat - xMean) / xVar
    yMean = np.mean(yMat, axis=0)  # 计算均值
    ryMat = yMat - yMean

    return rxMat, ryMat


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    函数说明：岭回归
    :param xMat:
    :param yMat:
    :param lam: 缩减系数
    :return:    回归系数
    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('矩阵为奇异阵， 不能求逆')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xMat, yMat):
    '''
    函数说明：岭回归测试，研究回归系数和缩减系数之间的关系
    :param xMat:
    :param yMat:
    :return:
    '''
    cols = np.shape(xMat)[1]
    yMean = np.mean(yMat, axis=0)  # 计算均值
    yMat = yMat - yMean
    xMean = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)    # 按列计算方差
    xMat = (xMat - xMean) / xVar
    numlam = 30                    # 缩减系数个数
    wsMat = np.zeros((numlam, cols))  # 初始回归系数矩阵
    for i in range(numlam):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wsMat[i,:] = ws.T

    return wsMat

def lossError(yMat, yHat):
    '''
    函数说明：计算真实值和预测值之间的误差
    :param yMat: 真实值
    :param yHat: 预测值
    :return:
    '''
    totalLoss = ((yMat.A - yHat) ** 2).sum()
    print('totalLoss:', totalLoss)

def stageWise(xMat, yMat, eps=0.01, index = 100):
    rxMat, ryMat = regularize(xMat, yMat)   #数据归一化
    rows, cols = np.shape(xMat)
    wsMat = np.zeros((index, cols))         # 初始化回归系数矩阵
    ws = np.zeros((cols, 1))                # 初始化回归系数
    wsTest = ws.copy()
    wsMax = ws.copy


def show_wsMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 数据集所在地址：
    path = 'D:/python/machine_learning_algorithm/Regression/DateSet/abalone.txt'
    data = loadDataSet(path)
    xMat = np.mat(data[:, 0:-1])  # 特征列
    yMat = np.mat(data[:, -1]).T  # 标签列，注意这里array转matrix时列向量会变行向量，一定要注意
    ridgeWeights = ridgeTest(xMat, yMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    title = ax.set_title(r'$\log(\lambda)$与回归系数$\omega$的关系', FontProperties=font)
    xlabel = ax.set_xlabel(r'$\log(\lambda)$', FontProperties=font)
    ylabel = ax.set_ylabel(u'回归系数$\omega$', FontProperties=font)
    plt.setp(title, size=15, weight='bold', color='red')
    plt.setp(xlabel, size=10, weight='bold', color='black')
    plt.setp(ylabel, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    show_wsMat()