# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:34:10 2018

@author: GEAR
"""

import operator
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

"""
函数说明:kNN算法,分类器

Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def KNNClassify(TestData, TrainData, Trainlabels, k):
    '''
    TestData是你要输入的要分类的“坐标”，TrainData是上面createDataSet的array，
    就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里
    面的k
    '''
    dataSetRows = TrainData.shape[0]  # 计算TrainData有多少行
    '''
    下面用tile，把一行TestData变成4行一模一样的（tile有重复的功能，
    #dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一
    行里有四个一样的）， 然后再减去dataSet，是为了求两点的距离
    ，先要坐标相减，这个就是坐标相减
    '''
    TempDist = np.tile(TestData, (dataSetRows, 1)) - TrainData  #
    SqTempDist = TempDist ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    SqDistances = SqTempDist.sum(axis=1)  # axis=1是行相加，这样得到了(x1-x2)^2+(y1-y2)^2
    distances = SqDistances ** 0.5
    '''
    #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0]) 
    '''
    sortedDistIndex = distances.argsort()
    classCout = {}

    '''
    operator.itemgetter(1),按照classCout的值进行排序，key=operator.itemgetter(0)根据字典的键进行
    排序reverse默认是false升序，这里设置为True，降序排列,dict.get(key, default=None)函数，key就是
    dict中的键voteIlabel，如果不存在则返回一个0，并存入dict，python3中用items()替换python2中的iteritems()
    '''
    for i in range(k):
        voteIlabel = Trainlabels[sortedDistIndex[i]]
        classCout[voteIlabel] = classCout.get(voteIlabel, 0)+1
    sortedClassCout = sorted(classCout.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCout[0][0]

"""
函数说明:将读取的32x32的二进制图像矩阵转换为1x1024数组。

Parameters:
	dir - 读取文件所在路径
Returns:
	returnVect - 返回的二进制图像的1x1024数组
"""

def imageVector(dir):
    # 创建1x1024数组
    vector = np.zeros((1, 1024))
    # 打开读取的文件
    fr = open(dir)
    # 按行读取文件
    templist = []
    for line in fr.readlines():
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        templine = list(map(float, line))
        templist.append(templine)        
        tempvector = np.array(templist)
        # 将多维数组转化为一维数组
        vector = tempvector.flatten()
        
    return vector

"""
函数说明:手写数字分类

Parameters:
    trainDir - 训练数据所在文件夹路径
    testDir - 测试数据所在文件夹路径
Returns:
	无
"""

def handwritingClassify(trainDir, testDir):

    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir(trainDir)
    # 返回文件夹下文件的个数
    trainNumbers = len(trainingFileList)
    # 初始化训练集的mat矩阵
    trainingDatas = np.zeros((trainNumbers,1024))
    # 训练集的labels
    trainLabels = []
    # 从文件名中提取出训练集的类别
    for i in range(trainNumbers):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类数字的标签
        classLabel = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        trainLabels.append(classLabel)
        # 将每一个文件的1x1024数组存储到trainingData中
        trainingDatas[i,:] = imageVector(trainDir + fileNameStr)
        
    # 返回testDigits目录下的文件名
    testFileList = listdir(testDir)
    # 错误检查计数
    errorCount = 0.0
    # 测试数据数量
    testNumbers = len(testFileList)
    # 初始化测试集的mat矩阵
    testDatas = np.zeros((testNumbers,1024))
    # 测试集的labels
    testLabels = []
    predictLabels = []
    
    for i in range(testNumbers):
        fileNameStr = testFileList[i]
        classLabel = int(fileNameStr.split('_')[0])
        testLabels.append(classLabel)
        testData = imageVector(testDir + fileNameStr)
        testDatas[i,:] = testData      
        # 获得预测结果
        classifierResult = KNNClassify(testData, trainingDatas, trainLabels, 3)
        predictLabels.append(classifierResult)
        print("分类结果为%d\t真实结果为%d" % (classifierResult, classLabel))
        if(classifierResult != classLabel):
            errorCount += 1.0
    print("\n分类错误个数%d\n错误率为%.3f%%" % (errorCount, (errorCount/testNumbers)*100))
    
    return trainingDatas, testDatas, trainLabels, testLabels, predictLabels


def showImageData(images, reallabels, predictlabels, idx, num=10):
    fig = plt.gcf()
    # 设置显示图像大小
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx].reshape(32, 32), cmap = 'binary')
        title = 'label=' + str(reallabels[idx])
        if len(predictlabels) > 0:
            title += ', predict=' + str(predictlabels[i])
        # 设置显示标题
        ax.set_title(title, fontsize=10)
        # 设置不显示刻度
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
           
if __name__ == '__main__':
    
    trainDir = 'D:/machine learing/py_code/KNN/KNN_mnist/trainingDigits/'
    testDir = 'D:/machine learing/py_code/KNN/KNN_mnist/testDigits/'
    trainingDatas, testDatas, trainLabels, testLabels, predictLabels = \
    handwritingClassify(trainDir, testDir)
    
    showImageData(testDatas, testLabels, predictLabels, 0, num=10)
    
    

        