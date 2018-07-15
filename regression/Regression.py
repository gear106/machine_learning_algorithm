# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:38:42 2018

@author: GEAR
"""

import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(path):
    '''
    函数说明：读取数据集
    :param path: 数据集文件所在路径
    :return:     数据集矩阵
    '''
    #逐行读取数据集
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
        dataMat[index,:] = line[:]
        index += 1
    return dataMat

def showDataSet(data):
    '''
    函数说明：显示数据集
    :param path: 数据集路径
    :return: none
    '''
    features = data[:,1]  # 特征列
    labels = data[:,2]    # 标签列
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
    xHat = xMat.copy()
    xHat.sort(axis=0)
    yHat = xHat * ws
    plt.figure(2)
    plt.plot(xHat[:,1], yHat, c='red')
    plt.scatter(xMat[:,1].A, yMat.A, s=20, c='blue', alpha=.5)
    plt.title('Regression')
    plt.xlabel('xValue')
    plt.ylabel('yValue')
    plt.show()


if __name__ == '__main__':
    # 数据集所在地址：
    path = 'D:/python/mechine_learning_algorithm/Regresssion/DateSet/regression.txt'
    data = loadDataSet(path)
    xMat = np.mat(data[:,0:2])  # 特征列
    yMat = np.mat(data[:,2]).T    # 标签列
    showDataSet(data)
    showRegression(xMat, yMat)



