# -*- coding: utf-8 -*-
"""
@Project : mente_carlo_tree_search_in_pedigreeTree
@File    : utils.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/15 20:30
"""
import re
import numpy as np
def readNex(path):
    info=[]
    speciesname=[]
    with open(path, "r") as f:  # 打开文件
        flag = 0
        # data =
        split_data=f.read().split('\n')# 读取文件
        begin = []
        end = []
        for i in split_data:
            if 'MATRIX' in i or 'matrix' in i or 'Matrix' in i :
                flag = 1
                begin.append(i)
                continue
            elif ';' == i.replace(" ",""):
                flag = 2
                end.append(i)
                continue
            if flag == 0:
                begin.append(i)
            elif flag == 2:
                end.append(i)
            elif flag == 1 and i != '':
                i = i.replace('\t', ' ')
                speciesname.append(i.strip().split(' ')[0])
                info.append(''.join(i.strip().split(' ')[1:]).replace("\t", ""))
    data=[[] for i in range(len(info))]
    for i in range(len(info)):
        for j, val in enumerate(info[i]):
            try:
                data[i].append(int(val))
            except:
                if val == "-":
                    data[i].append(-1)
                elif val == "?":
                    data[i].append(np.nan)
    misss_row = []
    for ind, i in enumerate(data):
        if np.nan in i:
            misss_row.append(ind)
            continue
    return np.array(data,dtype=float),misss_row,speciesname,begin,end

def hasP(index,i):
    """
    判断当前位置字符一直往做，是否会遇到P
    :param index:
    :param i:
    :return:
    """
    while (index>=0):
        if i[index]=="p":
            return False
        elif i[index]=="(":
            return True
        elif i[index]==",":
            return True
        index-=1
    return True

def replace_char(old_string, char, index,old_len):
    '''
    字符串按索引位置替换字符
    '''
    old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    new_string = old_string[:index] + char + old_string[index+old_len:]
    return new_string
def genera2tre(genera, speciesName):
    """
    数字广义表转为物种名广义表
    :param genera: 数字广义表
    :param spicename: 物种名列表
    :return:

    """
    for j in range(len(speciesName) - 1, -1, -1):
        t = [loc.start() for loc in re.finditer(str(j), genera)]
        for index in [loc.start() for loc in re.finditer(str(j), genera)]:
            if hasP(index, genera):
                genera = replace_char(genera, '{}'.format(speciesName[j]), index, len(str(j)))
                break
    return genera