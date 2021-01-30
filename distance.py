# -*- coding: utf-8 -*-
"""
@Project : mente_carlo_tree_search_in_pedigreeTree
@File    : distance.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/21 16:33
"""
#对分类单元的距离进行计算
import numpy as np
from utils import readNex
def hammingDistance(seq1,seq2):
    """
    :param data: 两个序列
    :return:序列的汉明距离，不计算含有缺失特征，不可适用作为新特征
    """
    if len(seq1) != len(seq2):
        raise ValueError("不等长")
    length =len(seq1)
    count=0
    for i,j in zip(seq1,seq2):
        if np.isnan(i) or np.isnan(j):
            length-=1
        elif i!=j:
           count+=1
    return count/length

def hammingDistance_missing(seq1,seq2):
    """
    :param data: 两个序列
    :return:序列的汉明距离，计算含有缺失特征，不可适用作为新特征
    """
    if len(seq1) != len(seq2):
        raise ValueError("不等长")
    length =len(seq1)
    count=0
    for i,j in zip(seq1,seq2):
        if i!=j:
           count+=1
    return count/length
def hammingMatrix(data):
    col=len(data)
    matrix=np.zeros((col,col))
    for i in range(col):
        for j in range(col):
            if i>j:
                matrix[i][j]=matrix[j][i]
            elif i<j:
                matrix[i][j]=hammingDistance(data[i],data[j])
    return matrix

def branchSum(distanceMatrix):
    """
    计算任意两点合并后的分支总和，NJ法选择最小值合并，
    蒙特卡洛法这里前几名都随机选择，一定概率全局随机选择。
    :param distanceMatrix: 距离矩阵
    :return: 两点合并分支距离矩阵
    """
    #所有分类单元距离之和
    T=np.sum(distanceMatrix)/2
    #分类单元数
    m=len(distanceMatrix)
    #任意两单元和并的S值矩阵，越小表示越接近
    SMatrix=np.zeros((m,m))
    #R为当前一个节点与其他所有节点的距离和
    R=np.zeros(m)
    for i in range(m):
        R[i]=sum(distanceMatrix[i])
    for i in range(m):
        for j in range(m):
            if i==j:
                SMatrix[i][j]=float('inf')
            elif i<j:
                SMatrix[i][j]=(2*T-R[i]-R[j])/(2*(m-2))+distanceMatrix[i][j]/2
            else:
                SMatrix[i][j] = float('inf')
    return SMatrix

def getNewDistanceMatrix(distanceMatrix,x,y):
    """
    根据选中的合并点，得到合并后的新距离
    :param matrix: 初始距离矩阵
    :param x: 选择合并的i点
    :param y: 选择合并的j点
    :return:合并后新的SMatrix矩阵
    """
    m=len(distanceMatrix)
    newDistanceMatrix=np.zeros((m-1,m-1))
    bigIndex=max(x,y)
    litterIndex=min(x,y)
    tmpDistanceMatrix=np.delete(distanceMatrix,bigIndex,axis=0)
    tmpDistanceMatrix =np.delete(tmpDistanceMatrix,litterIndex,axis=0)

    tmpDistanceMatrix =np.delete(tmpDistanceMatrix,bigIndex,axis=1)
    tmpDistanceMatrix =np.delete(tmpDistanceMatrix,litterIndex,axis=1)
    for i in range(m-2):
        for j in range(m-2):
            newDistanceMatrix[i][j]=tmpDistanceMatrix[i][j]
    for k in range(m-2):
        # 多次出现在某一位的距离为0
        newDistanceMatrix[-1][k]=newDistanceMatrix[k][-1]=(distanceMatrix[x][k]+distanceMatrix[y][k]-distanceMatrix[x][y])/2
    return newDistanceMatrix
def getNewTree(tree,x,y):
    """
    合并树，并返回新的结构
    :param tree:
    :param x:
    :param y:
    :return:
    """
    nameX=tree[x]
    nameY=tree[y]
    tree.remove(nameX)
    tree.remove(nameY)
    tree.append("({},{})".format(nameX,nameY))
    return tree







