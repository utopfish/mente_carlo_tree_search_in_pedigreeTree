# @Time:2019/11/20 14:13
# @Author:liuAmon
# @e-mail:utopfish@163.com
# @File:graph.py


class Node:
    __slots__ = ['id', 'neighbors', 'state']

    def __init__(self, id, neighbors, state):
        self.id = id
        self.state = state
        self.neighbors = neighbors


# 图的宽度遍历和深度遍历

# 1. BFS
def bfsTravel(graph, source):
    # 传入的参数为邻接表存储的图和一个开始遍历的源节点
    frontiers = [source]  # 表示前驱节点
    travel = [source]  # 表示遍历过的节点
    # 当前驱节点为空时停止遍历
    while frontiers:
        nexts = []  # 当前层的节点（相比frontier是下一层）
        for frontier in frontiers:
            for current in graph[frontier]:  # 遍历当前层的节点
                if current not in travel:  # 判断是否访问过
                    travel.append(current)  # 没有访问过则入队
                    nexts.append(current)  # 当前结点作为前驱节点
        frontiers = nexts  # 更改前驱节点列表
    return travel

def neighborHasVal(graph,current):
    '''
    获取邻居的val值
    :param graph:
    :param current:
    :return: [[]]
    '''
    count=[]
    for i in graph[current]:
        if val[i]!=[]:
            count.append(val[i])
    return count
def hasOnlyInapp(graphVal):
    for i in graphVal:
        if '-' in i and len(i)==1:
            return True
    return False
def deleteOneInapp(graphVal):
    for i in graphVal:
        if '-' in i and len(i)>1:
            i.remove('-')
    return graphVal
def IntersectionSet(graph):
    '''
    有交取交，无交取并
    :param graph:
    :return:
    '''
    temp=set(graph[0]).intersection(*graph[1:])
    if len(temp)==0:
        temp=set(graph[0]).union(*graph[1:])
    return list(temp)
def dfsTravelFirst(val,graph, source):
    # 传入的参数为邻接表存储的图和一个开始遍历的源节点
    travel = []  # 存放访问过的节点的列表
    stack = [source]  # 构造一个堆栈
    while stack:  # 堆栈空时结束
        current = stack.pop()  # 堆顶出队
        if current not in travel:  # 判断当前结点是否被访问过
            if len(val[current])==1:
                travel.append(current)
            elif val[current]==[]:
                temp=neighborHasVal(graph,current)
                if len(temp)>1:
                    if hasOnlyInapp(temp):
                        pass
                    else:
                        temp=deleteOneInapp(temp)
                    val[current]=IntersectionSet(temp)
                    travel.append(current)  # 如果没有访问过，则将其加入访问列表
        print(current,val[current])
        for next_adj in graph[current]:  # 遍历当前结点的下一级
            if next_adj not in travel:  # 没有访问过的全部入栈
                stack.append(next_adj)
    return travel,val

if __name__ == "__main__":
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
    graph['p'] = ['n', 'm', 'p']
    graph['q'] = ['l', 'm', 'r']
    graph['r'] = ['q', 'g', 's']
    graph['s'] = ['r', 'h', 'v']
    graph['t'] = ['i', 'u', 'v']
    graph['u'] = ['t', 'j', 'k']
    graph['v'] = ['s', 't']

    val={}
    val['a']=['0']
    val['b']=['1']
    val['c']=['-']
    val['d']=['-']
    val['e']=['-']
    val['f']=['-']
    val['g']=['1']
    val['h']=['-']
    val['i']=['-']
    val['j']=['-']
    val['k']=['2']
    val['l']=[]
    val['m']=[]
    val['n']=[]
    val['p']=[]
    val['q']=[]
    val['r']=[]
    val['s']=[]
    val['t']=[]
    val['u']=[]
    val['v']=[]

    # test of BFS
    # print(bfsTravel(graph, 'a'))

    print(dfsTravelFirst(val,graph, 'a'))
