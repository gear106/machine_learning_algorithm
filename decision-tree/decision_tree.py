# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 23:08:34 2018

@author: GEAR
"""

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle
from collections import Counter


def creatDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '工作', '房子', '信贷情况']  # 特征标签
    return dataSet, labels


def shannonEntropy(dataSet):
    numEntires = len(dataSet)  # 数据行数
    labelCounts = {}  # 存储每个特征出现的次数
    for feature in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = feature[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 记录每个特征出现的次数
        labelCounts[currentLabel] += 1  # 当前特征对应次数加一
    # 对于 label 标签的占比，求出 label 标签的香农熵
    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        entropy -= prob * log(prob, 2)
    return entropy


def splitDataSet(dataSet, axis, value):
    '''
    函数说明：通过遍历dataset，求出axis对应的列的值作为value的行
    最后得到含有value的行的所有数据，并且剔除value的值

    :param dataSet:
    :param axis:
    :param value:
    :return:
    '''
    retDataSet = []
    for feature in dataSet:
        if feature[axis] == value:
            reducedFeatVec = feature[:axis]  # 二维列表中的一行数据提取
            # [axis+1:]表示从跳过 axis 的 axis+1行，取接下来的数据
            reducedFeatVec.extend(feature[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    函数说明：通过计算信息增益，返回最好特征划分的索引
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1  # 计算第一行有多少列
    baseEntropy = shannonEntropy(dataSet)  # 原始信息熵
    bestInfoGain, bestFeature = 0.0, -1  # 最优的信息增益值和最优的特征编号
    for i in range(numFeatures):
        # 获取对应特征下的所有数据，将dataSet中每一列的数据都提取出来组成新的列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 使用set对list中的元素去重
        newEntropy = 0.0  # 创建一个新的临时的熵
        # 遍历当前特征中所有唯一的属性，对每个唯一属性划分一次数据集，计算数据集的新熵，
        # 并对所有唯一特征值得到的熵求和：
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))  # 计算该特征下某种情况的概率
            newEntropy += prob * shannonEntropy(subDataSet)  # 计算信息熵
        infoGain = baseEntropy - newEntropy  # 计算每个特征的信息增益
        print('infoGain=', infoGain, 'bestFeature=', i)
        if (infoGain > bestInfoGain):  # 找到最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt1(classList):
    '''
    函数说明：选择出现次数最多的一个结果
    :param classList:  labels列的集合
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒序排列classCount得到一个字典集合，然后取出最后一个就是结果（yes/no),即出现次数最多的结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 出现次数最多的元素


def majorityCnt2(classList):
    major_label = Counter(classList).most_common(1)[0]
    return major_label

def creatTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]    #获取分类标签（是否贷款）
    #count()函数统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList): #若类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:            #若分类特征只有一种时
        return majorityCnt1(classList)  #遍历所有特征时返回出现次数最多的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)    #选择最优特征
    bestFeatLabel = labels[bestFeat]                #选择最优特征标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}     #根据最优特征生成树的根节点
    del(labels[bestFeat])           #删除已经使用的特征标签
    featValues = [example[bestFeat] for example in dataSet] #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)    #去掉重复的属性值
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))    # 获取决策树的结点
    secondDict = inputTree[firstStr]    # 获取第二分支
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    '''
    函数说明：存储决策树
    :param inputTree: 已经生成的决策树
    :param filename: 决策树的存储文件名
    :return:
    '''
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    '''
    函数说明：读取存储的决策树
    :param filename:决策树存储文件名
    :return:        决策树字典
    '''
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    print(dataSet)
    entropy = shannonEntropy(dataSet)
    print(entropy)
    print('最优特征索引：' + str(chooseBestFeatureToSplit(dataSet)))
    featLabels = []
    myTree = creatTree(dataSet, labels, featLabels)
    storeTree(myTree, 'classifierStorage.txt')
    print(myTree)
    testVec = [0,1] #表示没房子有工作
    result = classify(myTree, featLabels, testVec)
    print(result)
