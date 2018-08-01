# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:37:35 2018

@author: GEAR
"""

from sklearn import svm
import numpy as np

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
函数说明:分类器测试函数

Parameters:
    dir - 训练数据路径
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
"""
def ClassTest(DataDir, LabelDir):
    # 将返回的特征值和分类向量分别储存到DataMat和Labels

    ImageDataMat= filematrix(DataDir, 4096, 1)
    ImageLabels = filematrix(LabelDir, 1, 2)
    # 取所有数据得百分之80%
    hoRatio = 0.8
    # 数据归一化，返回归一化后的值
    normMat = ImageDataMat
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之三十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    Cout = 0.0
        
    clf = svm.SVC(decision_function_shape = 'ovo')
    clf.fit(normMat[numTestVecs:m, :], ImageLabels[numTestVecs:m])

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为分类集，后m-numTestVecs个数据集作为训练集
        classifierResult = clf.predict(normMat[i, :].reshape(1,-1))
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
    # 训练和测试SVM模型
    ClassTest(DataDir, LabelDir)
    