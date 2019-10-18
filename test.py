# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test.py
#@time: 2019/10/17 22:44
import numpy as np
import pandas as pd
from mente_carlo_tree_search import is_terminal
import math
def divis(S):
    D = np.sum(S)
    result=[]
    for x in range(2 ** (len(S) - 1)):
        part1 = 0
        part2 = 0
        s1 = []
        s2 = []
        for i in range(len(S)):
            if (x >> i) % 2 == 1:
                part1 += S[i]
                s1.append(S[i])
            else:
                s2.append(S[i])
                part2 += S[i]
        if abs(part2 - part1) < D:
            result.append([s1,s2])
    return result

def readDataTxt(path):
    data=pd.read_table(path,header=None,sep=" ")
    return data
path2 = r"F:\实验室谱系树一切相关\谱系树软件\自研代码\singleCharacter-Fitch验证数据集\002号数据集\缺失数据集.txt"
data = readDataTxt(path2)
li = np.array(data)
k=10
initTree= [i for i in range(k)]
t=divis(initTree)
for i in t:
    print("------")
    print(i)
    print(is_terminal(i))
print(len(t))
print(int(math.pow(2,k-1)-2))







