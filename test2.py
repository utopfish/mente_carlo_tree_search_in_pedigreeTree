# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test2.py
#@time: 2019/10/17 23:05
import time
from tqdm import tqdm
import numpy as np
from mente_carlo_tree_search import get_children2,divis,readDataTxt,getFict,default_policy,is_terminal,get_children

path = r"F:\实验室谱系树一切相关\谱系树软件\自研代码\singleCharacter-Fitch验证数据集\005号最简化叶足动物\缺失数据集.txt"
data = readDataTxt(path)
li=np.array(data)

def analysis(test,num):
    while is_terminal(test)==False:
        s=get_children(test)
        for i in s:

            print("   "*num,end="")
            print("{}:{}".format(i, default_policy(i)))
            analysis(i,num+1)
        break



# for i in divis(test):
#     print("{}:{}".format(i,default_policy(i)))
if __name__=="__main__":
    test = [0, 1, 2, 3, 4]
    analysis(test,0)
    print(default_policy("((0,1),(2,(3,4)))"))