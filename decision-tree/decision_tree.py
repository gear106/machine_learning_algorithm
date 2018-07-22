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
    numEntires = len(dataSet)
    labelCounts = {}
    for feature in dataSet:
        currentLabel = feature[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        entropy -= prob * log(prob, 2)
    return entropy

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for feature in dataSet:
        if feature[axis] == value:
            reducedFeatVec = feature[:axis] #二维列表中的一行数据提取
            reducedFeatVec.extend(feature[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

            


if __name__ == '__main__':
    dataSet, features = creatDataSet()
    print(dataSet)
    entropy = shannonEntropy(dataSet)
    print(entropy)
    creatDataSet()



    