# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test2.py
#@time: 2019/10/17 23:05
import time
from tqdm import tqdm
from mente_carlo_tree_search import get_children2,divis2

test=  [[3], [0, 11, 13]]
test2=[0,11,13]
for i in range(100):
    print(divis2(test2))
