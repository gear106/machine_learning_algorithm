# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 19:53:10 2018

@author: GEAR
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


def loadDataSet1(path):
    '''
    函数说明:将导入的数据转化为one-hot数据
    :param path: 数据载入路径
    :return: 数据特征值和数据标签
    '''
    with open(path) as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 此数据总共有5列
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', 'target']
        lenses_pd = pd.DataFrame(lenses)
        lenses_pd.columns = lensesLabels
        lenses_target = list(lenses_pd.target)  # 将最后一列取出来赋值转换为列表
        lenses_pd = lenses_pd.drop('target', axis=1)  # 去掉最后一列

        le = LabelEncoder()
        for col in lenses_pd.columns:
            lenses_pd[col] = le.fit_transform(lenses_pd[col])  # 将字典对应的字符转换为数字

    return  lenses_pd, lenses_target

def sklearnDecisionTree(features, labels):
    '''
    函数说明: 采用sklearn自带的decision_tree方法对数据进行分类预测
    :param features: 数据特征值
    :param labels:   数据标签值
    :return:
    '''
    clf = tree.DecisionTreeClassifier(max_depth=4)  # 建立决策树类
    clf = clf.fit(features.values.tolist(), labels)  # 使用实际数据创建决策树模型
    print(clf.predict([[1, 1, 1, 0]]))  # 采用决策树模型做预测


if __name__ == '__main__':
    path = 'D:/python/machine_learning_algorithm/decision-tree/dataSet/lenses.txt'
    features, labels = loadDataSet1(path)
    sklearnDecisionTree(features, labels)