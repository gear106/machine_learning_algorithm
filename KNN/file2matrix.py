# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:07:24 2018

@author: GEAR
"""
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# pyhton3中画图显示中文会有问题，通过载人计算机中的字体解决

def file2matrix(filename):
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, 2))
    index = 0
        
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split()
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        index += 1
          
    return returnMat
  
# 读入数据路径
dir = 'D:/CFD2018/naca0012/x_y.dat'
returnMat = file2matrix(dir)
x = returnMat[:,0]
y = returnMat[:,1]
plt.plot(x,y)
plt.grid()
plt.show()
