# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:07:24 2018

@author: GEAR
"""
import operator
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# pyhton3中画图显示中文会有问题，通过载人计算机中的字体解决

"""
函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

Parameters:
    dir - 训练数据存放路径
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
"""
def file2matrix(dir):
    fr = open(dir)
    # 读取文件所有内容
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类
        classLabelVector.append(listFromLine[-1])
        index += 1

    return returnMat, classLabelVector


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
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classify0(TestData, TrainData, Trainlabels, k):
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
函数说明:分类器测试函数

Parameters:
    dir - 训练数据路径
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
"""
def datingClassTest(dir):
    # 将返回的特征值和分类向量分别储存到datingDataMat和datingLabels
    datingDataMat, datingLabels = file2matrix(dir)
    # 取所有数据得百分之十
    hoRatio = 0.3
    # 数据归一化，返回归一化后的值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCout = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为分类集，后m-numTestVecs个数据集作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果：%s\t真实类别：%s" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCout += 1.0
        print("错误率：%f%%" %(errorCout/float(numTestVecs)*100))

"""
函数说明:可视化数据

Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类Label
Returns:
    无
"""
def showdatas(dataMat, datalabels):
    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10, 12))
    LabelsColors = []
    for i in datalabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')

    # 画出散点图， 以dataMat矩阵第一列，第二列为坐标，散点大小15，透明度0.5
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=dataMat[:, 0], y=dataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=dataMat[:, 0], y=dataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dataMat[:, 1], y=dataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()
"""
函数说明：输入一个人的三维特征，进行分类输出
Parameters:
    dir - 训练数据路径

"""
def classifyPerson(dir):
    #输入人的特征
    FlyMiles = float(input("每年获得飞行常客里程数："))
    GameTimes = float(input("玩视频游戏所耗时间百分百："))
    IceCream = float(input("每周消费冰淇淋公升数："))
    #打开并读入训练数据
    TrainDataMat, TrainLabels = file2matrix(dir)
    #训练数据归一化
    normTrainMat, ranges, minVals = autoNorm(TrainDataMat)
    #生成测试集
    TestMat = np.array([FlyMiles, GameTimes, IceCream])
    #测试集归一化
    normTestMat = (TestMat - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(normTestMat, normTrainMat, TrainLabels, 4)
    # 输出结果
    print("you might {} the person".format(classifierResult))

# 读入数据路径
"""
函数说明:main函数
"""
if __name__ == '__main__':
    dir = 'D:\machine learing\data\datingTestSet.txt'
    # 训练和测试K-mean模型
    datingClassTest(dir)
    # 对输入的数据进行预测
    classifyPerson(dir)
    # 显示输入的训练数据
#    datingDataMat, datingLabels = file2matrix(dir)
#    showdatas(datingDataMat, datingLabels)
