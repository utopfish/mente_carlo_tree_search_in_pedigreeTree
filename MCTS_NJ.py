# -*- coding: utf-8 -*-
"""
@Project : mente_carlo_tree_search_in_pedigreeTree
@File    : MCTS_NJ.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/21 19:05
"""
import sys
import copy
import time
import math
import random
import logging
import numpy as np
import pandas as pd
from singleCharacterFitch import getsingleFitchs
from utils import genera2tre

global treeResult
from distance import *
from NJ import probabilityChoice, KNeig, minLoc

treeResult = {}


# state表示在list中的str字符，len(state)>2表示还能继续合并，非叶子节点
# tree表示广义表
class Node(object):
    """
    蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
    """

    def __init__(self):
        self.parent = None
        self.children = []
        self.MaxchildrenNumber = 0
        self.visit_times = 0  # 访问次数
        self.quality_value = 0.0  # 得分
        self.fitchScore = 0
        self.score = None
        self.childrenPool = []  # chilidren的待选库
        self.is_all_explored = False

    def setState(self, state):
        self.state = state

    def getState(self):
        state = copy.deepcopy(self.state)
        return state

    def setDistanceMatrix(self, distanceMatrix):
        self.distanceMatrix = distanceMatrix

    def getDistanceMatrix(self):
        distanceMatrix = copy.deepcopy(self.distanceMatrix)
        return distanceMatrix

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_addN(self, n):
        self.quality_value += n

    def get_fitchScore(self):
        return self.fitchScore

    def fitchScore_addN(self, n):
        self.fitchScore += n

    def is_all_expand(self):
        if self.MaxchildrenNumber == 0:
            self.MaxchildrenNumber = find_canDivisNumber(self.state)
        return len(self.children) == self.MaxchildrenNumber

    def add_child(self, subNode):
        subNode.set_parent(self)
        self.children.append(subNode)

    def __repr__(self):

        C = 1 / math.sqrt(2.0)
        if self.parent != None:
            return "NodeScore: {},fitch:{}, Q/N: {}, visit_time:{},state: {}".format(
                self.score,
                self.fitchScore, self.quality_value / self.visit_times, self.visit_times, self.state)
        else:
            return "NodeScore: {},fitch:{}, Q/N: {}, visit_time:{},state: {}".format(
                self.score, self.fitchScore, self.quality_value / self.visit_times, self.visit_times, self.state)


def divis(S):
    # 返回随机划分的一个结果,对分类的结果按大小进行排序
    if len(S) < 3:
        return [S]
    while 1:

        part1 = 0
        part2 = 0
        s1 = []
        s2 = []
        for i in range(len(S)):
            x = random.choice([0, 1])
            if x % 2 == 1:
                part1 += 1
                s1.append(S[i])
            else:
                s2.append(S[i])
                part2 += 1

        if part1 != 0 and part2 != 0:
            s1.sort()
            s2.sort()
            if s1[0] < s2[0]:
                return [s1, s2]
            else:
                return [s2, s1]


def is_terminal(treeSatus):
    '''
    判断树结构是否继续可分
    :param treeSatus: str
    :return:
    '''
    if isinstance(treeSatus[0], int):
        if len(treeSatus) > 2 and isinstance(treeSatus[1], int):
            return False if len(treeSatus) > 2 else True
    re = True
    for index, i in enumerate(treeSatus):
        if isinstance(i, list):
            re = True if re and is_terminal(i) else False
    return re


def find_canDivisNumber(treeStatus):
    '''
    判断树结构所有可分组合数量
    :param treeStatus: list
    :return:
    '''
    sum = 1
    if isinstance(treeStatus[0], int):
        if len(treeStatus) > 2:
            return sum * int(math.pow(2, len(treeStatus) - 1) - 1)
    for index, i in enumerate(treeStatus):
        if isinstance(i, list):
            if is_terminal(i) == False:
                sum = sum * find_canDivisNumber(i)
    if sum == 1:
        return 0
    return sum


def find_canDivis(treeStatus):
    '''
    找到树结构中可进行继续划分的部分
    :param treeStatus:
    :return:
    '''
    if isinstance(treeStatus[0], int):
        if len(treeStatus) > 2:
            return [treeStatus]
    re = []
    for index, i in enumerate(treeStatus):
        if isinstance(i, list):
            if is_terminal(i) == False:
                re.append(i)
    return re


def getNextState_with_random_choice(treeSatus):
    '''
    随机获取下一个树结构
    :param treeSatus:
    :return:
    '''
    if is_terminal(treeSatus):
        return treeSatus
    if isinstance(treeSatus[0], int) and len(treeSatus) > 2:
        return random.choice(divis(treeSatus))
    newTree = copy.deepcopy(treeSatus)
    for index, i in enumerate(treeSatus):
        if isinstance(i, list):
            newTree[index] = getNextState_with_random_choice(i)
    return newTree


def get_children(treeStatus):
    '''
    获取随机一个子节点树结构
    :param treeStatus:
    :return:
    '''
    if is_terminal(treeStatus):
        if isinstance(treeStatus[0], int) and len(treeStatus) == 1:
            return treeStatus[0]
        return treeStatus
    elif isinstance(treeStatus[0], int) and len(treeStatus) > 2:
        return divis(treeStatus)
    else:
        newTree = copy.deepcopy(treeStatus)
        for index, i in enumerate(treeStatus):
            if isinstance(i, list):
                newTree[index] = get_children(i)
        return newTree


def getNextState_with_random_choice2(node):
    # 对分割结果进行存储
    if len(node.childrenPool):
        pass
    else:
        node.childrenPool = get_children(node.getState())
    return random.choice(node.childrenPool)


def getNextState_with_random_choic3(node):
    # 随机选择分割结果
    return get_children(node.getState())


def is_allexplored(node):
    if not node.is_all_expand():
        return False
    result = True
    for i in node.get_children():
        result = result and is_allexplored(i)
    node.is_all_explored = result
    return result


def state2Tree(state):
    """
    list转为广义表
    :param state:
    :return:
    """
    return "(" + ",".join(state) + ")"


class monte_carlo_treeSearch():
    """
    实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。
    蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
    前两步使用tree policy找到值得探索的节点。
    第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
    最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
    进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
    """

    def __init__(self, Node, data, computationBudget=10):
        """

        :param Node:
        :param computationBudget:
        :param data: 特征矩阵
        """
        self.node = Node
        self.computationBudget = computationBudget
        self.data = data

    def setNode(self, Node):
        self.node = Node

    def search(self):
        # Run as much as possible under the computation budget
        # 单轮最大迭代次数
        for i in range(self.computationBudget):
            # 1. Find the best node to expand
            expandNode = self.tree_policy(self.node)

            # 2. Random run to add node and get reward
            if expandNode == None:
                break
            reward = self.default_policy(expandNode.getState(), expandNode.getDistanceMatrix())

            # 3. Update all passing nodes with reward
            self.backup(expandNode, reward)

        # N. Get the best next node
        if self.node == None:
            return None
        bestNextNode = self.best_child(self.node, True)

        return bestNextNode

    def tree_policy(self, node):
        '''
        蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），
        根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
        基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，
        如果UCB值相等则随机选。
        :param node:
        :return:
        '''

        # Check if the current node is the leaf node
        while node != None and len(node.getState()) > 2:
            # 获取当前最好子节点
            node = self.best_child(node, True)
            # Return the new sub node
            subNode = self.expand(node)
            return subNode

        # Return the leaf node
        return node

    def default_policy(self, currentState, currentMatrix):
        """
        蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。
        基本策略是随机选择Action。
        """
        # # Get the state of the game
        while len(currentState) > 2:
            currentState, currentMatrix = self.getNextState_withNJ(currentState, currentMatrix)

        currentTree = state2Tree(currentState)
        treeScore = getsingleFitchs(currentTree, self.data)
        if treeScore not in treeResult:
            treeResult[treeScore] = [currentTree]
        elif currentTree not in treeResult[treeScore]:
            treeResult[treeScore].append(currentTree)
        return treeScore

    def backup(self, node, reward):
        """
            蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
            并将该节点与其父节点访问次数加一
            """
        # Update util the root node
        while node != None:
            # Update the visit times
            node.visit_times_add_one()

            # Update the quality value
            node.quality_value_addN(reward)

            # Change the node to the parent node
            node = node.parent

    def best_child(self, node, is_exploration):
        """
        使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
        使用得分减处理之后的访问次数进行选择
        """

        # TODO: Use the min float value
        bestScore = sys.maxsize
        bestSubNode = None

        # Travel all sub nodes to find the best one
        for subNode in node.get_children():

            # Ignore exploration for inference
            if is_exploration:
                C = 1 / math.sqrt(2.0)
            else:
                C = 0.0

            # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            left = subNode.get_quality_value() / subNode.get_visit_times()
            right = 5.0 * math.log(node.get_visit_times()) / subNode.get_visit_times()
            score = left - C * math.sqrt(right)
            # score = left

            if score < bestScore:
                bestSubNode = subNode
                bestScore = score

        if bestScore == sys.maxsize:
            bestSubNode = node
        return bestSubNode

    def expand(self, node):
        """
        输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
        """

        triedSubNodeStates = [
            subNode.getState() for subNode in node.get_children()
        ]

        newTreeState, newDistanceMatrix = self.getNextState_withNJ(node.getState(), node.getDistanceMatrix())
        # Check until get the new state which has the different action from others
        # 设定重复次数
        repeatTime = 0
        ##TODO：找到按概率随机选择数据
        while newTreeState in triedSubNodeStates and repeatTime < 100:
            newTreeState, newDistanceMatrix = self.getNextState_withNJ(node.getState(), node.getDistanceMatrix())
            repeatTime += 1
        subNode = Node()
        subNode.setState(newTreeState)
        subNode.setDistanceMatrix(newDistanceMatrix)
        node.add_child(subNode)

        return subNode

    ### 辅助函数

    def getNextState_withNJ(self, treeStatus, distanceMatix, type="pro"):
        SMatrix = branchSum(distanceMatix)
        # TODO:当前没有对结果进行探索(已经添加了随机选择)
        if type == "pro":
            x, y = probabilityChoice(SMatrix)
        elif type == "random":
            # 可以设置k值，默认为3
            x, y = KNeig(SMatrix, k=10)
        elif type == "normal":
            x, y = minLoc(SMatrix)
        matrix = getNewDistanceMatrix(distanceMatix, x, y)
        treeStatus = getNewTree(treeStatus, x, y)
        return treeStatus, matrix


def saveRet(savePath,treeResult,spicesName):
    allRet = set()
    for i in treeResult.keys():
        for tree in treeResult[i]:
            allRet.add(genera2tre(tree,spicesName))

    with open(savePath,'w') as f:
        for i in allRet:
            f.writelines(i+";\n")




def main(savePath,data, spicename):
    """
    :param data:特征数据集
    :return:
    """
    start = time.time()
    initTree = [str(i) for i in range(len(data))]
    distanceMatrix = hammingMatrix(data)
    currentNode = Node()
    currentNode.setState(initTree)
    currentNode.setDistanceMatrix(distanceMatrix)
    MCTS = monte_carlo_treeSearch(currentNode, data)

    count = 0
    global treeResult
    temp = sorted(treeResult.items())
    treeResult = {}
    while currentNode != None and count < 50:
        MCTS.setNode(currentNode)
        currentNode = MCTS.search()
        if currentNode == None:
            break
        print("Play round: {}".format(count + 1))
        count += 1
        print("Choose node: {}".format(currentNode))
        print(time.time() - start)
        # global treeResult
        temp = sorted(treeResult.items())
        treeResult = {}
        treeResult[temp[0][0]] = temp[0][1]
        print(treeResult)
        print("最短树长个数:{}".format(len(temp[0][1])))
        if is_terminal(currentNode.getState()) == True:
            while currentNode.parent != None:
                currentNode = currentNode.parent
    saveRet(savePath,treeResult,speciesname)


if __name__ == "__main__":
    # TODO 编写用例
    # TODO 使用list装最小树得分的树(已完成)
    # TODO 加速fitch算法
    # TODO 对部分不可枝进行提前删除，eg：通过距离，通过初始树
    # TODO 代码整理
    # TODO 对打分部分进行整理
    # TODO 加入Alpha-beta方法

    path = r"C:\Users\pro\Desktop\实验三蒙特卡洛树\真实数据集\Aria2015.nex"
    savePath=r'C:\Users\pro\Desktop\实验三蒙特卡洛树\蒙特卡洛建树结果\Aria2015.nex'
    data, misss_row, speciesname, begin, end = readNex(path)
    main(savePath,data, speciesname)
