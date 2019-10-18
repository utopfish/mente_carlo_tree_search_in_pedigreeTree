# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test2.py
#@time: 2019/10/17 23:05
from singleCharacterFitch import getSingleChararcterFitch
test="({2, 3, 4, 7, 15, 16, 19}, {0, 1, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 20})".replace(" ","")
character=[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
print(getSingleChararcterFitch(test,character))