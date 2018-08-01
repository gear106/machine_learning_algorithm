# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:07:24 2018

@author: GEAR
"""
import numpy as np
import matplotlib.pyplot as plt

"""
函数说明:对数据进行归一化

Parameters:
    fileName - 载入数据文件路径
Returns:
    dataSet - 返回数据矩阵
"""
def loadDataSet(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        curLine = line.split()
        # map函数对curLine列表中的数做数据类型转化
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return np.mat(dataSet)

"""
函数说明:计算两个向量的欧氏距离(Euclidean distance)

Parameters:
    vecA - 输入向量
    vecB - 输入向量
Returns:
    return - 返回l两个向量的欧氏距离
"""
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

"""
函数说明:随机生成k个初始聚类中心

Parameters:
    dataSet - 输入数据
    k - 输入聚类中心个数
Returns:
    centroids - 返回k个随机聚类中心的坐标
"""
def randCent(dataSet, k):
    colus = dataSet.shape[1]  #计算数据的列数
    centroids = np.mat(np.zeros((k, colus)))
    for j in range(colus):
        colusMin = np.min(dataSet[:, j]) # 找到第j列的最小值
        colusMax = np.max(dataSet[:, j]) # 找到第j列的最大值
        ranges = float(colusMax - colusMin)
        # 生成k行1列范围在（0,1）之间的随机数矩阵
        centroids[:, j] = colusMin + ranges*np.random.rand(k, 1)

    return centroids

def KMeans(dataSet, k, distMeans = distEclud, creatCent = randCent):
    rows = dataSet.shape[0] # 数据集的行数
    # 矩阵clusterAssment为rows行2列，该矩阵和dataSet的行数相同，行行对应，
    # 第一列表示与对应行距离最近的质心下标，第二列表示欧式距离的平方。
    clusterAssment = np.mat(np.zeros((rows, 2)))
    centroids = creatCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(rows):  #遍历数据集汇总的每一行数据
            minDist = np.inf; minIndex = -1
            for j in range(k):  #寻找最近的质心
                # 计算输入数据的每一行和初始生成的聚类中心的距离
                tempDist = distMeans(dataSet[i,:], centroids[j,:])
                # 若当前距离小于初始距离，更新最小距离和质心下标
                if tempDist < minDist:
                    minDist = tempDist; minIndex = j
            # 若当前数据点所对应的聚类中心不是距离最小的聚类中心，重新寻找
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        # 更新聚类中心位置
        for cent in range(k):
            # clusterAssment.A将mat转换为array
            # np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。
            # ptsInClust数组储存属于同一类的所有元素的数据
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


## step 1: load data
print("step 1: load data...")
filename = 'D:/machine learing/py_code/K-means/testSet.txt'
dataSet = loadDataSet(filename)

## step 2: clustering...
print("step 2: clustering...")
k = 4
centroids, clusterAssment = KMeans(dataSet, k)

## step 3: show the result
print("step 3: show the result...")
showCluster(dataSet, k, centroids, clusterAssment)










