# -*- coding: utf-8 -*-
"""
Created on Fri May 18 18:13:12 2018

@author: GEAR
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:07:24 2018

@author: GEAR
"""
import operator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

"""
函数说明:打开并解析文件

Parameters:
    dir - 训练数据存放路径
    colums - 每行数据总数
    tag - tag=1表示读取的是图像特征，tag=2表示读取的是对应标签
Returns:
    returnMat - 特征矩阵
    LabelVector - 分类Label向量
"""
def filematrix(dir, columns, tag):
    fr = open(dir)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, columns))
    LabelVector = []
    index = 0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        if tag == 1:
            returnMat[index,:] = listFromLine[0:columns]
        elif tag == 2:
            LabelVector.append(listFromLine[0])
        index += 1
    if tag == 1:
        return returnMat
    elif tag == 2:
        return LabelVector


"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
"""
def autoNorm(dataSet):
    # 获取数据最小值,dataAet.min(0)中的参数使得函数
    # 可以获取列中的最小值，而不是当前行的最小值
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回矩阵的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大值和最小值的差，得到归一化的数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果，数据范围，最小值
    return normDataSet, ranges, minVals
"""
函数说明:kNN算法,分类器

Parameters:
    TestData - 用于分类的数据
    TrainData - 用于训练的数据
    Trainlabels - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classify0(TestData, TrainData, Trainlabels, k):
    '''
    inX是你要输入的要分类的“坐标”，dataSet是上面createDataSet的array，
    就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里
    面的k
    '''
    dataSetSize = TrainData.shape[0]  # 计算dataSet有多少行
    '''
    下面用tile，把一行TestData变成4行一模一样的（tile有重复的功能，
    #dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一
    行里有四个一样的）， 然后再减去dataSet，是为了求两点的距离
    ，先要坐标相减，这个就是坐标相减
    '''
    diffMat = np.tile(TestData, (dataSetSize, 1)) - TrainData  #
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1是行相加，这样得到了(x1-x2)^2+(y1-y2)^2
    distances = sqDistances ** 0.5
    '''
    #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0]) 
    '''
    sortedDistIndicies = distances.argsort()
    classCout = {}

    '''
    operator.itemgetter(1),按照classCout的第二维进行排序，reverse默认是false升序，这里设置为True
    降序排列,dict.get(key, default=None)函数，key就是dict中的键voteIlabel，如果不存在则返回一个0
    并存入dict，python3中用items()替换python2中的iteritems()
    '''
    for i in range(k):
        voteIlabel = Trainlabels[sortedDistIndicies[i]]
        classCout[voteIlabel] = classCout.get(voteIlabel, 0) + 1
    sortedClassCout = sorted(classCout.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCout[0][0]


"""
函数说明:分类器测试函数

Parameters:
    dir - 训练数据路径
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
"""
def datingClassTest(DataDir, LabelDir):
    # 将返回的特征值和分类向量分别储存到datingDataMat和datingLabels

    ImageDataMat= filematrix(DataDir, 4096, 1)
    ImageLabels = filematrix(LabelDir, 1, 2)
    # 取所有数据得百分之十
    hoRatio = 0.4
    # 数据归一化，返回归一化后的值
    normMat = ImageDataMat
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类正确计数
    Cout = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为分类集，后m-numTestVecs个数据集作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     ImageLabels[numTestVecs:m], 10)
        print("分类结果：%s\t真实类别：%s" %(classifierResult, ImageLabels[i]))
        if classifierResult == ImageLabels[i]:
            Cout += 1.0
        print("正确率：%f%%" %(Cout/float(numTestVecs)*100))

"""
函数说明:main函数
"""
if __name__ == '__main__':
    DataDir = r"ImageFc7Mat.txt"  #分类图像fc7特征值
    LabelDir = r"ImageLabels.txt"  #分类图像标签
    # 训练和测试K-mean模型
    datingDataMat= filematrix(DataDir, 4096, 1)
    datingLabels = filematrix(LabelDir, 1, 2)
    datingClassTest(DataDir, LabelDir)

        

