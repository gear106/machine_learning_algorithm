# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:39:51 2018

@author: GEAR
"""
import numpy as np
import pandas as pd


def writeToTxt1(array,file_path):
    try:
        fp = open(file_path,"w+")
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                fp.write(str(array[i][j]) + '\t')
            fp.write('\n')
        fp.close()
    except IOError:
        print("fail to open file")
        
def writeToTxt2(labels,file_path):
    try:
        fp = open(file_path,"w+")
        for item in labels:
            fp.write(str(item)+"\n") # list中一项占一行
        fp.close()
    except IOError:
        print("fail to open file")
        
def filematrix(filename, columns, tag):
    fr = open(filename)
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
 
if __name__ == "__main__":
    labels = ['aa', 'bb']
 
    array = np.array([[3,2,1],[1,0,4]])
    file_path1 = r'D:/machine learing/py_code/K-means/testSet.txt'
    file_path2 = r"ImageLabels.txt"
    #writeToTxt1(array,file_path)
    a = filematrix(file_path1, 2, 1)
    b = filematrix(file_path2, 1, 2)
    
    