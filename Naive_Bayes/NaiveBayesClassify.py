# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:43:29 2018

@author: GEAR
"""

import numpy as np
from functools import reduce

def loadDataSet():
    #切分的词条
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的列表
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        # 若词条存在于词汇表中，则其将其在词汇表中对应位置的值设为1, 
        # vocabList.index(word)返回word在列表中的索引值
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not my vocabulary!" % word)
    return returnVec

def creatVocabList(dataSet):
    # 创建一个空的不重复的集合，python集合里不能有重复的元素
    vocabSet = set([])
    for document in dataSet:
        # 取并集，将dataSet中所有的单词存到一个不重复的集合中
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def NaiveBays(trainMatrix, trainCategory):
    # 统计训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 计算词汇表中的词汇总数
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率，#类别标签1代表侮辱性词汇，0代表不是
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    # 词条出现初始化为0数组
#    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
#    p0Denom = 0.0; p1Denom = 0.0
    # 这里为防止几所多概率乘积时出现一个概率值为0，最后的乘积
    # 为的情况，将上述代码修改为如下：
    # 统计每个单词在相应类别下出现的条件概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    # 统计在所给中出现侮辱性和非侮辱性词条向量的总次数
    p0Denom = 1.0; p1Denom = 1.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
        # p(w|1) = p(w&1)/p(1),sum(trainMatrix[i])计算在此类别下所有单词出现
        # 的总次数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)...
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0Vect, p1Vect, pAbusive

def NativeBaysClassify(vec2Classify, p0Vec, p1Vec, pClass1):
    #  reduce(lambda x, y: x+y, [1,2,3,4,5])  # 使用 lambda 匿名函数
    # 计算列表和：1+2+3+4+5,在 Python3 中，reduce() 函数已经被从全局
    # 名字空间里移除了，它现在被放置在 fucntools 模块里，如果想要使用
    # 它，则需要通过引入 functools 模块来调用 reduce() 函数：
    p1 = reduce(lambda x,y: x*y, vec2Classify * p1Vec)  + np.log(pClass1)
    p0 = reduce(lambda x,y: x*y, vec2Classify * p0Vec) +  np.log(1 - pClass1)
    
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testNativebaysClassify():
    # 生成所要统计文档的词汇列表和对应的类别
    listPosts, listClasses = loadDataSet()
    # 统计文档中所有的词汇并生成词汇向量
    myVocabList = creatVocabList(listPosts)
    trainMat = []
    # 统计文档中词汇在词汇表中的位置
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0Vec, p1Vec, pAb = NaiveBays(np.array(trainMat), np.array(listClasses))
    testDoc1 = ['love','my','dalmation']
    # 将测试文档转换为词汇向量形式，统计其中每个单词在词汇表中出现的次数和对应的位置
    thisDoc1 = np.array(setOfWords2Vec(myVocabList, testDoc1))
    if  NativeBaysClassify(thisDoc1, p0Vec, p1Vec, pAb):
        print(testDoc1,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testDoc1,'属于非侮辱类')										#执行分类并打印分类结果
    testDoc2 = ['stupid', 'garbage']										#测试样本2

    thisDoc2 = np.array(setOfWords2Vec(myVocabList, testDoc2))				#测试样本向量化
    if  NativeBaysClassify(thisDoc2, p0Vec, p1Vec, pAb):
        print(testDoc2,'属于侮辱类')										#执行分类并打印分类结果
    else:
        print(testDoc2,'属于非侮辱类')	
    
            
            
    
    


if __name__ == '__main__':
    testNativebaysClassify()

    