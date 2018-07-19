# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 02:08:51 2018

@author: GEAR
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def loadDataSet1(path1, path2):
    '''
    函数说明：读取训练和测试数据
    :param path1: 训练数据路径
    :param path2: 测试数据路径
    :return: 训练数据和测试数据的features / labels
    '''
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
    trainArr = trainData[:, 0:-1]  # 取出trainData除最后一列的所有数据
    trainLab = trainData[:, -1].astype(np.int)  # 取出trainData的最后一列
    testArr = testData[:, 0:-1]
    testLab = testData[:, -1].astype(np.int)

    return trainArr, trainLab, testArr, testLab


def loadDataSet2(path1, path2):
    '''
    函数说明：同loadDataSet1函数，这里采用pandas库简化读取数据编码工作量
    :param path1: 训练数据路径
    :param path2: 测试数据路径
    :return: 训练数据和测试数据的features / labels
    '''
    trainData = pd.read_table(path1, header=None)
    testData = pd.read_table(path2, header=None)
    trainArr = trainData.iloc[:, 0:21].values               # 训练数据
    trainLab = trainData.iloc[:, -1].values.astype(np.int)  # 训练标签
    testArr = testData.iloc[:, 0:21].values                 # 测试数据
    testLab = testData.iloc[:, -1].values.astype(np.int)    # 测试标签
    return trainArr, trainLab, testArr, testLab


def classifyTest(trainArr, trainLab, testArr, testLab):
    '''
    函数说明：采用sklearn库的logistics回归函数做二元分类
    :param trainArr: 训练数据features
    :param trainLab: 训练数据labels
    :param testArr:  测试数据features
    :param testLab:  测试数据labels
    :return:         测试数据预测labels
    '''
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainArr, trainLab)
    testPredict = classifier.predict(testArr)
    testAccurcy = classifier.score(testArr, testLab) * 100
    print('测试集正确率为：%.2f%%' % testAccurcy)
    return testPredict


if __name__ == '__main__':
    path1 = 'D:/python/machine_learning_algorithm/logistic-regression/DataSet/horseColicTraining.txt'
    path2 = 'D:/python/machine_learning_algorithm/logistic-regression/DataSet/horseColicTest.txt'
    # trainArr, trainLab, testArr, testLab = loadDataSet1(path1, path2)  # 读取训练数据和测试数据
    trainArr, trainLab, testArr, testLab = loadDataSet2(path1, path2)  # 读取训练数据和测试数据
    testPredict = classifyTest(trainArr, trainLab, testArr, testLab)