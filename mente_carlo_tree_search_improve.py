#@Time:2020/8/13 20:06
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:mente_carlo_tree_search_improve.py
__author__ = "liuAmon"

import sys
import copy
import time
import math
import random
import logging
import numpy as np
import pandas as pd
from singleCharacterFitch import getsingleFitchs
from fitch import getFitchs
from utils import readNex
from singleCharacterFitch import readDataTxt
from sklearn.cluster import KMeans
global treeResult
treeResult = {}



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
        self.fitch_score = 0
        self.score = None
        self.childrenPool = []  # chilidren的待选库
        self.is_all_explored = False

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

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

    def quality_value_add_n(self, n):
        self.quality_value += n

    def get_fitch_score(self):
        return self.fitch_score

    def fitch_score_add_n(self, n):
        self.fitch_score += n

    def is_all_expand(self):
        if self.MaxchildrenNumber == 0:
            self.MaxchildrenNumber = find_can_divis_number(self.state)
        return len(self.children) == self.MaxchildrenNumber

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):

        C = 1 / math.sqrt(2.0)
        if self.parent != None:
            return "NodeScore: {},fitch:{}, Q/N: {}, visit_time:{},state: {}".format(
                self.score,
                self.fitch_score, self.quality_value / self.visit_times, self.visit_times, self.state)
        else:
            return "NodeScore: {},fitch:{}, Q/N: {}, visit_time:{},state: {}".format(
                self.score, self.fitch_score, self.quality_value / self.visit_times, self.visit_times, self.state)


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
            x = random.choice([0,1])
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


def find_can_divis_number(treeStatus):
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
                sum = sum * find_can_divis_number(i)
    if sum == 1:
        return 0
    return sum


def find_can_divis(treeStatus):
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





def get_next_state_with_random_choice(treeSatus):
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
            newTree[index] = get_next_state_with_random_choice(i)
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



def get_next_state_with_random_choice2(node):
    # 对分割结果进行存储
    if len(node.childrenPool):
        pass
    else:
        node.childrenPool = get_children(node.get_state())
    return random.choice(node.childrenPool)


def get_next_state_with_random_choic3(node):
    # 随机选择分割结果
    return get_children(node.get_state())


def is_allexplored(node):
    if not node.is_all_expand():
        return False
    result = True
    for i in node.get_children():
        result = result and is_allexplored(i)
    node.is_all_explored = result
    return result



class monte_carlo_tree_search():
    """
    实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。
    蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
    前两步使用tree policy找到值得探索的节点。
    第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
    最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
    进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
    """
    def __init__(self,Node,data,computation_budget=100):
        """

        :param Node:
        :param computation_budget:
        :param data: 特征矩阵
        """
        self.node=Node
        self.computation_budget=computation_budget
        self.data=data
    def set_node(self,Node):
        self.node=Node
    def search(self):
        # Run as much as possible under the computation budget
        for i in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(self.node)

            # 2. Random run to add node and get reward
            if expand_node == None:
                break
            reward = self.default_policy(expand_node.get_state())

            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)

        # N. Get the best next node
        if self.node == None:
            return None
        best_next_node = self.best_child(self.node, True)

        return best_next_node
    def tree_policy(self,node):
        '''
        蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
        基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
        :param node:
        :return:
        '''

        # Check if the current node is the leaf node
        while node != None and is_terminal(node.get_state()) == False:
            # 判断是否可能的结果全部探索完，这里可以用来限制探索的广度
            if node.is_all_expand():
                node = self.best_child(node, True)
            else:
                # Return the new sub node
                sub_node = self.expand(node)
                return sub_node

        # Return the leaf node
        return node
    def default_policy(self,current_state):
        """
        蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。
        基本策略是随机选择Action。
        """
        # Get the state of the game
        while is_terminal(current_state) == False:
            current_state = get_children(current_state)
        res = find_can_divis(current_state)
        current_state = str(current_state)
        for i in res:
            current_state = current_state.replace(str(i), str(i).replace("[", "{").replace("]", "}"))
        current_state = current_state.replace("[", "(").replace("]", ")").replace(" ", "")
        treeScore = getsingleFitchs(current_state, self.data)
        if treeScore not in treeResult:
            treeResult[treeScore] = [current_state]
        elif current_state not in treeResult[treeScore]:
            treeResult[treeScore].append(current_state)
        return treeScore

    def backup(self,node, reward):
        """
            蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
            并将该节点与其父节点访问次数加一
            """
        # Update util the root node
        while node != None:
            # Update the visit times
            node.visit_times_add_one()

            # Update the quality value
            node.quality_value_add_n(reward)

            # Change the node to the parent node
            node = node.parent

    def best_child(self,node, is_exploration):
        """
        使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
        使用得分减处理之后的访问次数进行选择
        """

        # TODO: Use the min float value
        best_score = sys.maxsize
        best_sub_node = None

        # Travel all sub nodes to find the best one
        for sub_node in node.get_children():

            # Ignore exploration for inference
            if is_exploration:
                C = 1 / math.sqrt(2.0)
            else:
                C = 0.0

            # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            left = sub_node.get_quality_value() / sub_node.get_visit_times()
            right = 5.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
            score = left - C * math.sqrt(right)
            # score = left

            sub_node.score = score
            if not is_allexplored(sub_node) and score < best_score:
                best_sub_node = sub_node
                best_score = score

        if best_score == sys.maxsize:
            best_sub_node = node.parent
        return best_sub_node

    def expand(self,node):
        """
        输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
        """

        tried_sub_node_states = [
            sub_node.get_state() for sub_node in node.get_children()
        ]

        # new_state = get_next_state_with_random_choice3(node)
        new_state = self.get_next_state_with_e_kMean(node.get_state())
        # Check until get the new state which has the different action from others
        # while new_state in tried_sub_node_states:
        #     new_state = get_next_state_with_random_choice3(node)
        while new_state in tried_sub_node_states:
            new_state = self.get_next_state_with_e_kMean(node.get_state())

        sub_node = Node()
        sub_node.set_state(new_state)
        node.add_child(sub_node)

        return sub_node
    ### 辅助函数

    def divis_kmean(self,S):
        '使用kmean对数据进行二分类,并将结果排序'
        if len(S) < 3:
            return [S]
        temp = []
        for s in S:
            temp.append(self.data[s])
        temp = np.array(temp)
        kmeans_model = KMeans(n_clusters=2).fit(temp)
        labels = kmeans_model.labels_
        part1 = []
        part2 = []
        for index, s in enumerate(S):
            if labels[index] == 0:
                part1.append(s)
            else:
                part2.append(s)
        part1.sort()
        part2.sort()
        if part1[0] < part2[0]:
            return [part1, part2]
        else:
            return [part2, part1]

    def get_next_state_with_e_kMean(self,treeStatus, theta=0.9):
        # kmean 聚类结果,0.5概率随机，0.5概率kmean

        if is_terminal(treeStatus):
            if isinstance(treeStatus[0], int) and len(treeStatus) == 1:
                return treeStatus[0]
            return treeStatus
        elif isinstance(treeStatus[0], int) and len(treeStatus) > 2:
            if random.random() > theta:
                return self.divis_kmean(treeStatus)
            else:
                return divis(treeStatus)
        else:
            newTree = copy.deepcopy(treeStatus)
            for index, i in enumerate(treeStatus):
                if isinstance(i, list):
                    newTree[index] = self.get_next_state_with_e_kMean(i)
            return newTree
def readDataTxt(path):
    data = pd.read_table(path, header=None, sep=" ")
    return data


def main(data):
    """
    :param data:特征数据集
    :return:
    """
    start = time.time()
    # Create the initialized state and initialized node

    initTree = [i for i in range(len(data))]

    init_node = Node()
    init_node.set_state(initTree)
    MCTS = monte_carlo_tree_search(init_node,data)
    current_node=MCTS.search()


    count = 0
    print("Play round: {}".format(count))
    print("Choose node: {}".format(current_node))
    global treeResult
    temp = sorted(treeResult.items())
    treeResult = {}
    treeResult[temp[0][0]] = temp[0][1]
    print(treeResult)
    print("最短树长个数:{}".format(len(temp[0][1])))
    # print("最短树长个数:{}".format(len(temp[0][1])))
    while current_node != None:
        MCTS.set_node(current_node)
        current_node = MCTS.search()
        if current_node == None:
            break
        print("Play round: {}".format(count + 1))
        count += 1
        print("Choose node: {}".format(current_node))
        print(time.time() - start)
        # global treeResult
        temp = sorted(treeResult.items())
        treeResult = {}
        treeResult[temp[0][0]] = temp[0][1]
        print(treeResult)
        print("最短树长个数:{}".format(len(temp[0][1])))
        if is_terminal(current_node.get_state()) == True:
            while current_node.parent != None:
                current_node = current_node.parent


if __name__ == "__main__":
    # TODO 编写用例
    # TODO 使用list装最小树得分的树(已完成)
    # TODO 加速fitch算法
    # TODO 对部分不可枝进行提前删除，eg：通过距离，通过初始树
    # TODO 代码整理
    # TODO 对打分部分进行整理
    # TODO 加入Alpha-beta方法

    path = r"C:\Users\pro\Desktop\实验三蒙特卡洛树\真实数据集"
    data, misss_row, speciesname, begin, end=readNex(path)
    main(data)