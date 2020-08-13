# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test2.py
#@time: 2019/10/17 23:05
import time
from tqdm import tqdm
import numpy as np
import time
from mente_carlo_tree_search import *

path = r"C:\liuAmon_core_code\mente_carlo_tree_search_in_pedigreeTree\testData\011号简化数据集奇虾\011号完整数据集.txt"
data = readDataTxt(path)
li=np.array(data)

def analysis(test,num,result):
    while is_terminal(test)==False:
        s=get_children(test)
        for i in s:
            print("   "*num,end="")
            treeScore=default_policy(i)
            print("{}:{}".format(i, treeScore))

            if is_terminal(i):
                if treeScore not in result:
                    result[treeScore]=[i]
                elif i not in result[treeScore]:
                    result[treeScore].append(i)
            analysis(i, num + 1, result)
        break
    return result



# for i in divis(test):
#     print("{}:{}".format(i,default_policy(i)))
if __name__=="__main__":
    a='(a,(b,c))'
    b='((b,c),a)'
