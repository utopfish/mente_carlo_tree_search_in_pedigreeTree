#@Time:2019/11/27 17:44
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:test_graph.py
import pytest
from graphWork import *

graph = {}
graph['a'] = ['l']
graph['b'] = ['l']
graph['c'] = ['m']
graph['d'] = ['n']
graph['e'] = ['p']
graph['f'] = ['p']
graph['g'] = ['r']
graph['h'] = ['s']
graph['i'] = ['t']
graph['j'] = ['u']
graph['k'] = ['u']
graph['l'] = ['a', 'b', 'q']
graph['m'] = ['c', 'q', 'n']
graph['n'] = ['d', 'm', 'p']
graph['p'] = ['n', 'e', 'f']
graph['q'] = ['l', 'm', 'r']
graph['r'] = ['q', 'g', 's']
graph['s'] = ['r', 'h', 'v']
graph['t'] = ['i', 'u', 'v']
graph['u'] = ['t', 'j', 'k']
graph['v'] = ['s', 't']

val = {}
val['a'] = ['0']
val['b'] = ['1']
val['c'] = ['-']
val['d'] = ['-']
val['e'] = ['-']
val['f'] = ['0']
val['g'] = ['1']
val['h'] = ['-']
val['i'] = ['-']
val['j'] = ['-']
val['k'] = ['2']
val['l'] = []
val['m'] = []
val['n'] = []
val['p'] = []
val['q'] = []
val['r'] = []
val['s'] = []
val['t'] = []
val['u'] = []
val['v'] = []
def test_IntersectionSet():
    test = [['2'], ['5'], ['1']]
    assert set(['1','5','2'])==set(IntersectionSet(test))
def test_travelAll():
    global val,graph
    for i in val.keys():
        travel, val = (TravelFirst(val, graph, i))
        val=TravelSecond(travel[::-1],graph,val)
        test_val={'a': ['0'], 'b': ['1'], 'c': ['-'], 'd': ['-'], 'e': ['-'], 'f': ['0'], 'g': ['1'], 'h': ['-'], 'i': ['-'], 'j': ['-'], 'k': ['2'], 'l': ['1'], 'm': ['-'], 'n': ['-'], 'p': ['-'], 'q': ['1'], 'r': ['1'], 's': ['-'], 't': ['-'], 'u': ['-'], 'v': ['-']}
        assert test_val==val
if __name__=="__main__":

    pytest.main()