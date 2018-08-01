# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:45:31 2018

@author: GEAR
"""
import numpy as np

"""
函数说明:对数据进行归一化

Parameters:
    fileName - 载入数据文件路径
Returns:
    dataSet - 返回数据矩阵
"""
def loadDataSet(fileName, columns):
    dataSet = []
    f = open(fileName)
    arrayOLines = f.readlines()
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, columns))
    index = 0
    for line in f.readlines():
        curLine1 = line.strip()
        curLine = curLine1.split('\t')
        returnMat[index,:] = curLine[0:columns]
        index += 1
    return returnMat



if __name__ == "__main__":
    dir = r"testSet.txt"
    dataSet = []
    f = open(dir)
    for line in f.readlines():
        # map函数对curLine列表中的数做数据类型转化
        line = line.split()
        fltLine = list(map(float, line))
        dataSet.append(fltLine)
        dataMat = np.mat(dataSet)
