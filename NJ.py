# -*- coding: utf-8 -*-
"""
@Project : mente_carlo_tree_search_in_pedigreeTree
@File    : NJ.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/21 18:47
"""
import heapq
import random
from distance import *
def minLoc(SMatrix):
    """
    选中最小值对应位点，i,j
    :param SMatrix: 分支和距离
    :return:
    """
    x,y=np.where(SMatrix == np.min(SMatrix))
    return x[0],y[0]

def KNeig(SMatrix,k=3):
    li=[]
    m=len(SMatrix)
    k=min(k,m)
    for i in range(m):
        for j in range(i,m):
            li.append((-SMatrix[i][j],i,j))
    ret=heapq.nlargest(k,li,key=lambda x:x[0])
    branch,x,y=random.choice(ret)
    return x,y
def probabilityChoice(SMatrix):
    li = []
    m = len(SMatrix)
    for i in range(m):
        for j in range(i, m):
            li.append((-SMatrix[i][j], i, j))
    ret = heapq.nlargest(m, li, key=lambda x: x[0])
    indexs=[i for i in range(m)]
    pro = np.array([1/(i+1) for i in range(m)])
    pro=pro/sum(pro)

    index= np.random.choice(indexs,p = pro.ravel())
    branch, x, y=ret[index]
    return x, y
def NJ(data,speciesname):
    """
    缺失数据忽略，不可适用作为新特征处理
    :param data: 特征矩阵
    :param speciesname: 物种名
    :return:
    """
    tree = speciesname
    matrix = hammingMatrix(data)
    while (len(tree) > 2):
        SMatrix = branchSum(matrix)
        x, y = minLoc(SMatrix)
        matrix = getNewDistanceMatrix(matrix, x, y)
        tree = getNewTree(tree, x, y)
    tree = getNewTree(tree, 0, 1)
    return tree[0]

def NJRandom(data,speciesname):
    """
    缺失数据忽略，不可适用作为新特征处理,随机选中排名前k的结合点
    :param data: 特征矩阵
    :param speciesname: 物种名
    :return:
    """
    tree = speciesname
    matrix = hammingMatrix(data)
    while (len(tree) > 2):
        SMatrix = branchSum(matrix)
        x, y = KNeig(SMatrix)
        matrix = getNewDistanceMatrix(matrix, x, y)
        tree = getNewTree(tree, x, y)
    tree = getNewTree(tree, 0, 1)
    return tree[0]


def NJPro(data,speciesname):
    """
    缺失数据忽略，不可适用作为新特征处理,按排名分之一的概率进行随机抽取
    :param data: 特征矩阵
    :param speciesname: 物种名
    :return:
    """
    tree = speciesname
    matrix = hammingMatrix(data)
    while (len(tree) > 2):
        SMatrix = branchSum(matrix)
        x, y = probabilityChoice(SMatrix)
        matrix = getNewDistanceMatrix(matrix, x, y)
        tree = getNewTree(tree, x, y)
    tree = getNewTree(tree, 0, 1)
    return tree[0]
if __name__=="__main__":
    path = r"C:\Users\pro\Desktop\实验三蒙特卡洛树\真实数据集\Dikow2009.nex"
    data, misss_row, speciesname, begin, end=readNex(path)
    tree=NJPro(data,speciesname)
    print(tree)