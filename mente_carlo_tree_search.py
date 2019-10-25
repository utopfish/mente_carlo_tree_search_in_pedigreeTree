# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:mente_carlo_tree_search.py
#@time: 2019/10/17 21:32
import sys
import time
import math
import random
import copy
import numpy as np
import pandas as pd
from singleCharacterFitch import getFict
AVAILABLE_CHOICES = [1, -1, 2, -2]
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_ROUND_NUMBER = 10
import logging
from singleCharacterFitch import readDataTxt
path2 = r"F:\实验室谱系树一切相关\谱系树软件\自研代码\singleCharacter-Fitch验证数据集\002号数据集\缺失数据集.txt"
data = readDataTxt(path2)

li = np.array(data)
class State(object):
  """
  蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
  需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
  """

  def __init__(self):
    self.current_value = 0.0
    # For the first root node, the index is 0 and the game should start from 1
    self.current_round_index = 0
    self.cumulative_choices = []

  def get_current_value(self):
    return self.current_value

  def set_current_value(self, value):
    self.current_value = value

  def get_current_round_index(self):
    return self.current_round_index

  def set_current_round_index(self, turn):
    self.current_round_index = turn

  def get_cumulative_choices(self):
    return self.cumulative_choices

  def set_cumulative_choices(self, choices):
    self.cumulative_choices = choices

  def is_terminal(self):
    # The round index starts from 1 to max round number
    return self.current_round_index == MAX_ROUND_NUMBER

  def compute_reward(self):
    return -abs(1 - self.current_value)

  def get_next_state_with_random_choice(self):
    random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])

    next_state = State()
    next_state.set_current_value(self.current_value + random_choice)
    next_state.set_current_round_index(self.current_round_index + 1)
    next_state.set_cumulative_choices(self.cumulative_choices +
                                      [random_choice])

    return next_state

  def __repr__(self):
    return "State: {}, value: {}, round: {}, choices: {}".format(
        hash(self), self.current_value, self.current_round_index,
        self.cumulative_choices)


class Node(object):
  """
  蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
  """

  def __init__(self):
    self.parent = None
    self.children = []
    self.MaxchildrenNumber=0
    self.visit_times = 0 #访问次数
    self.quality_value = 0.0 #得分
    self.childrenPool=[] #chilidren的待选库


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

  def is_all_expand(self):
      if self.MaxchildrenNumber==0:
          self.MaxchildrenNumber=find_can_divis_number(self.state)
      return len(self.children) == self.MaxchildrenNumber

  def add_child(self, sub_node):
    sub_node.set_parent(self)
    self.children.append(sub_node)

  def __repr__(self):
    return "Node: {}, Q/N: {}/{}, state: {}".format(
        hash(self), self.quality_value, self.visit_times, self.state)


def divis(S):
    D = np.sum(S)
    result=[]
    if len(S)<3:
        return [S]
    for x in range(2 ** (len(S) - 1)):
        part1 = 0
        part2 = 0
        s1 = []
        s2 = []
        for i in range(len(S)):
            if (x >> i) % 2 == 1:
                part1 += S[i]
                s1.append(S[i])
            else:
                s2.append(S[i])
                part2 += S[i]
        if abs(part2 - part1) < D:
            result.append([s1,s2])
    return result
def divis2(S):
    #返回一个结果
    D = np.sum(S)
    result=[]
    if len(S)<3:
        return [S]
    while 1:
        x=random.choice(range(2 ** (len(S) - 1)))
        part1 = 0
        part2 = 0
        s1 = []
        s2 = []
        for i in range(len(S)):
            if (x >> i) % 2 == 1:
                part1 += S[i]
                s1.append(S[i])
            else:
                s2.append(S[i])
                part2 += S[i]
        if abs(part2 - part1) < D:
            return [s1,s2]


def is_terminal(treeSatus):
    if isinstance(treeSatus[0],int):
        if len(treeSatus)>2 and isinstance(treeSatus[1],int):
            return False if len(treeSatus)>2 else True
    re = True
    for index,i in enumerate(treeSatus):
        if isinstance(i,list):
            re= True if re and is_terminal(i) else False
    return re
def find_can_divis_number(treeStatus):
    sum=1
    if isinstance(treeStatus[0],int):
        if len(treeStatus)>2:
            return sum*int(math.pow(2,len(treeStatus)-1)-2)
    for index,i in enumerate(treeStatus):
        if isinstance(i,list):
            if is_terminal(i)==False:
                sum=sum*find_can_divis_number(i)
    return sum
def find_can_divis(treeStatus):
    if isinstance(treeStatus[0],int):
        if len(treeStatus)>2:
            return [treeStatus]
    re = []
    for index,i in enumerate(treeStatus):
        if isinstance(i,list):
            if is_terminal(i)==False:
                re.append(i)
    return re


def tree_policy(node):
  """
  蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。
  基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
  """

  # Check if the current node is the leaf node
  while is_terminal(node.get_state()) == False:
        #判断是否可能的结果全部探索完，这里可以用来限制探索的广度
    if node.is_all_expand():
      node = best_child(node, True)
    else:
      # Return the new sub node
      sub_node = expand(node)
      return sub_node

  # Return the leaf node
  return node

def get_next_state_with_random_choice(treeSatus):
    if is_terminal(treeSatus):
        return treeSatus
    if  isinstance(treeSatus[0],int) and len(treeSatus)>2:
        return random.choice(divis(treeSatus))
    newTree=copy.deepcopy(treeSatus)
    for index,i in enumerate(treeSatus):
        if isinstance(i,list):
            newTree[index]=get_next_state_with_random_choice(i)
    return newTree
def get_children(treeStatus):
    if is_terminal(treeStatus):
        return [treeStatus]
    if isinstance(treeStatus[0], int) and len(treeStatus) > 2:
        return divis(treeStatus)
    newTree = copy.deepcopy(treeStatus)
    resultTreeNow = []
    resultTreeNow.append(newTree)
    for index, i in enumerate(treeStatus):
        if isinstance(i, list):
            resultTreePre=copy.deepcopy(resultTreeNow)
            resultTreeNow=[]
            for tree in resultTreePre:
                for k in get_children(i):
                    newtr=copy.deepcopy(tree)
                    newtr[index] = k
                    resultTreeNow.append(newtr)
    return resultTreeNow
def get_children2(treeStatus):
    if is_terminal(treeStatus):
        if isinstance(treeStatus[0],int) and len(treeStatus)==1:
            return treeStatus[0]
        return treeStatus
    elif  isinstance(treeStatus[0],int) and len(treeStatus)>2:
        return divis2(treeStatus)
    else:
        newTree=copy.deepcopy(treeStatus)
        for index,i in enumerate(treeStatus):
            if isinstance(i,list):
                newTree[index]=get_children2(i)
        return newTree

def get_next_state_with_random_choice2(node):
    #对分割结果进行存储
    if len(node.childrenPool):
        pass
    else:
        node.childrenPool=get_children(node.get_state())
    return random.choice(node.childrenPool)
def get_next_state_with_random_choice3(node):
    #随机选择分割结果
    return get_children2(node.get_state())




def default_policy(node):
  """
  蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。
  基本策略是随机选择Action。
  """

  # Get the state of the game
  current_state = node.get_state()
  res=find_can_divis(current_state)
  current_state=str(current_state)
  for i in res:
      current_state =current_state.replace(str(i),str(i).replace("[","{").replace("]","}"))
  current_state =current_state.replace("[","(").replace("]",")").replace(" ","")

  return getFict(current_state,li)



def expand(node):
  """
  输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
  """

  tried_sub_node_states = [
      sub_node.get_state() for sub_node in node.get_children()
  ]

  new_state = get_next_state_with_random_choice3(node)

  # Check until get the new state which has the different action from others
  while new_state in tried_sub_node_states:
    new_state = get_next_state_with_random_choice3(node)

  sub_node = Node()
  sub_node.set_state(new_state)
  node.add_child(sub_node)

  return sub_node


def best_child(node, is_exploration):
  """
  使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
  """

  # TODO: Use the min float value
  best_score = -sys.maxsize
  best_sub_node = None

  # Travel all sub nodes to find the best one
  for sub_node in node.get_children():

    # Ignore exploration for inference
    if is_exploration:
      C = 1 / math.sqrt(2.0)
    else:
      C = 0.0

    # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
    left = -sub_node.get_quality_value() / sub_node.get_visit_times()
    right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
    # score = left + C * math.sqrt(right)
    score=left

    if score > best_score:
      best_sub_node = sub_node
      best_score = score

  return best_sub_node


def backup(node, reward):
  """
  蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
  """

  # Update util the root node
  while node != None:
    # Update the visit times
    node.visit_times_add_one()

    # Update the quality value
    node.quality_value_add_n(reward)

    # Change the node to the parent node
    node = node.parent


def monte_carlo_tree_search(node):
  """
  实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。
  蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
  前两步使用tree policy找到值得探索的节点。
  第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
  最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。
  进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
  """

  computation_budget = 200

  # Run as much as possible under the computation budget
  for i in range(computation_budget):

    # 1. Find the best node to expand
    expand_node = tree_policy(node)

    # 2. Random run to add node and get reward

    reward = default_policy(expand_node)

    # 3. Update all passing nodes with reward
    backup(expand_node, reward)


  # N. Get the best next node
  best_next_node = best_child(node, False)

  return best_next_node

def readDataTxt(path):
    data=pd.read_table(path,header=None,sep=" ")
    return data
def main():
  start=time.time()
  # Create the initialized state and initialized node
  path2 = r"F:\实验室谱系树一切相关\谱系树软件\自研代码\singleCharacter-Fitch验证数据集\005号最简化叶足动物\缺失数据集.txt"
  data = readDataTxt(path2)
  li = np.array(data)
  initTree=[ i for i in range(len(li))]
  # initTree = [i for i in range(6)]
  print(initTree)

  init_node = Node()
  init_node.set_state(initTree)
  current_node =monte_carlo_tree_search(init_node)
  # Set the rounds to play
  logging.basicConfig(filename="out-{}.log".format(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())), level=logging.WARNING)

  count=0
  print("Play round: {}".format(count))
  print("Choose node: {}".format(current_node))
  while current_node!=None:

      current_node = monte_carlo_tree_search(current_node)
      print("Play round: {}".format(count + 1))
      count += 1
      print("Choose node: {}".format(current_node))
      print(time.time() - start)
      if is_terminal(current_node.get_state())==True:
          logging.warning("round: {}".format(count+1))
          logging.warning("result:"+str(current_node.get_state()).replace("[","(").replace("]",")"))
          logging.warning("fitch treeScore:{}".format(default_policy(current_node)))
          logging.warning("visit times :{}".format(current_node.get_visit_times()))
          logging.warning("cost time: {} s".format(int(time.time()-start)))
          logging.warning("------------------------------------")

          while current_node.parent!=None:
              current_node=current_node.parent



if __name__ == "__main__":
  main()