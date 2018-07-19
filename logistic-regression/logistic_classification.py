# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 00:16:27 2018

@author: GEAR
"""

import numpy as np

'''
采用logistic regression 来做一个关于病马死亡率的预测
病针对常规梯度上升法和随机梯度上升法做了比较
'''

def readDataSet(path1, path2):
    trainData = [];testData = []
    with open(path1) as trainFr:
        for line in trainFr.readlines():
            line = line.strip().split()
            trainData.append(line)
    with open(path2) as testFr:
        for line in testFr.readlines():
            line = line.strip().split()
            testData.append(line)

    trainData = np.array(trainData).astype(np.float32)
    testData = np.array(testData).astype(np.float32)
    trainArr = trainData[:,0:-1]    #取出trainData除最后一列的所有数据
    trainLab = trainData[:,-1].astype(np.int)      #取出trainData的最后一列
    testArr = testData[:,0:-1]
    testLab = testData[:,-1].astype(np.int)

    return trainArr, trainLab, testArr, testLab

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
    for i in range(maxIter):
        h = sigmoid(featsMat * weights)
        error = labelMat - h
        weights = weights + alpha * featsMat.T * error
    return weights.A.reshape(cols)#将weights由mat转换为array

def stocGradAscent1(features, labels, maxIter=150):
    featsMat = np.array(features)
    labelMat = np.array(labels)
    rows, cols = np.shape(features)
    weights = np.ones((cols))       # 权重系数矩阵
    for j in range(maxIter):
        dataIndex = list(range(rows))
        for i in range(rows):
            alpha = 4 / (1.0 + i + j) + 0.01    #减小学习速率，每次减小1/(i+j)
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(featsMat[randIndex]*weights))
            error = labelMat[randIndex] - h
            weights = weights + alpha * error * featsMat[randIndex]
            del(dataIndex[randIndex])   #删除数据
    return  weights

def sigmoid(x):
    result = 1.0 / (1 + np.exp(-x))
    return result

def classifyVector(features, weights):
    prob = sigmoid(sum(features * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def classifyTest(trainArr, trainLab, testArr, testLab):
#    trainWeights = stocGradAscent1(trainArr, trainLab, maxIter=500) # 改进的随机梯度上升法
    trainWeights = gradAscent(trainArr, trainLab, alpha=0.01, maxIter=500) # 改进的随机梯度上升法
    errorCount = 0
    for i in range(testArr.shape[0]):
        if classifyVector(testArr[i], trainWeights) != testLab[i]:
            errorCount += 1
    errorRate = (float(errorCount / testArr.shape[0])) * 100    #计算错误率
    print('测试集错误率为：%.2f%%' % errorRate)


if __name__ == '__main__':
    path1 = 'D:/python/machine_learning_algorithm/logistic-regression/DataSet/horseColicTraining.txt'
    path2 = 'D:/python/machine_learning_algorithm/logistic-regression/DataSet/horseColicTest.txt'
    trainArr, trainLab, testArr, testLab = readDataSet(path1, path2)   #读取训练数据和测试数据
    classifyTest(trainArr, trainLab, testArr, testLab)

