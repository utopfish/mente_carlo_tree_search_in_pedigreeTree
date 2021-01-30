# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test2.py
#@time: 2019/10/17 23:05
import time
from tqdm import tqdm
import numpy as np
import time
from distance import hammingDistance,hammingDistance_missing
def str2np(st):
    ret=np.zeros(len(st))
    for index,i in enumerate(st):
        if i=="?":
            ret[index]=np.nan
        else:
            ret[index]=int(i)
    return ret
if __name__=="__main__":
    sp1="13231031230101130331023001303112331123???????????????10111023210221202122101323233200113320223330003"
    sp2="33202113230211001312200032003212231221???????????????23312110121013122133310132021121300022113122032"
    sp1=str2np(sp1)
    sp2=str2np(sp2)
    print(hammingDistance_missing(sp1,sp2))
