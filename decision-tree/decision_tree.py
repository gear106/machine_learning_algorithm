# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:08:34 2018

@author: GEAR
"""

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

def creatDataSet():
    
    dataSet = [[0, 0, 0, 0, 'no'],						#数据集
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
    labels = ['年龄', '工作', '房子', '信贷情况']   #特征标签
    return dataSet, labels

def shannonEntropy(dataSet):
    numEntires = len(dataSet)   #数据行数
    labelCounts = {}            #存储每个特征出现的次数
    for feature in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = feature[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0           #记录每个特征出现的次数
        labelCounts[currentLabel] += 1              #当前特征对应次数加一
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
            reducedFeatVec = feature[:axis] #二维列表中的一行数据提取
            # [axis+1:]表示从跳过 axis 的 axis+1行，取接下来的数据
            reducedFeatVec.extend(feature[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    函数说明：通过计算信息增益，返回最好特征划分的索引
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1   #计算第一行有多少列
    baseEntropy = shannonEntropy(dataSet)   #原始信息熵
    bestInfoGain, bestFeature = 0.0, -1   #最优的信息增益值和最优的特征编号
    for i in range(numFeatures):
        #获取对应特征下的所有数据，将dataSet中每一列的数据都提取出来组成新的列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  #使用set对list中的元素去重
        newEntropy = 0.0    #创建一个新的临时的熵
        #遍历当前特征中所有唯一的属性，对每个唯一属性划分一次数据集，计算数据集的新熵，
        # 并对所有唯一特征值得到的熵求和：
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))    #计算该特征下某种情况的概率
            newEntropy += prob * shannonEntropy(subDataSet) #计算信息熵
        infoGain = baseEntropy - newEntropy                 #计算每个特征的信息增益
        print('infoGain=', infoGain, 'bestFeature=', i)
        if(infoGain > bestInfoGain):                         #找到最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature





            


if __name__ == '__main__':
    dataSet, features = creatDataSet()
    print(dataSet)
    entropy = shannonEntropy(dataSet)
    print(entropy)
    print('最优特征索引：' + str(chooseBestFeatureToSplit(dataSet)))



    