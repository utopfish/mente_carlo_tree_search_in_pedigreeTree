# @Time:2019/11/13 15:48
# @Author:liuAmon
# @e-mail:utopfish@163.com
# @File:fitch.py
import pandas as pd
import numpy as np
def readDataTxt(path):
    data = pd.read_table(path, header=None, sep=" ")
    return data
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.visited = False
        self.tracker = None

    def set_val(self, val):
        self.val = val

    def get_val(self):
        return self.val

    def set_left(self, left):
        self.left = left

    def get_left(self):
        return self.left

    def set_right(self, right):
        self.right = right

    def get_right(self):
        return self.right

    def is_visited(self):
        return self.visited

    def visit(self):
        self.visited = True

    def clear_visit(self):
        self.visited = False


class Tree:
    def __init__(self, general_lists):
        self.root = self.findMid(general_lists)
        self.allElement = []
        self.score = [0]

    def findMid(self, general_lists):
        # 二叉分类
        mark = 0
        bigmark = 0
        mid = 0
        if general_lists[0] == "{":
            return Node(set(general_lists[1:-1].split(",")))
        else:
            for index, i in enumerate(general_lists[1:-1]):
                if (i == "("):
                    mark += 1
                elif (i == ")"):
                    mark -= 1
                elif i == "{":
                    bigmark += 1
                elif i == "}":
                    bigmark -= 1
                elif (i == "," and mark == 0 and bigmark == 0):
                    mid = index
                    break
            if mid == 0:
                return Node({general_lists})
            else:
                root = Node(set())
                root.left = self.findMid(general_lists[1:mid + 1])
                root.right = self.findMid(general_lists[mid + 2:-1])
                return root

    def postorderGetAllElement(self, root):
        if (root is None):
            return
        self.postorderGetAllElement(root.left)
        self.postorderGetAllElement(root.right)
        for i in root.val:
            if i != "?":
                self.allElement.append(i)

    def postorderInit(self, root, stateList):
        if (root is None):
            return
        self.postorderInit(root.left, stateList)
        self.postorderInit(root.right, stateList)
        if len(root.val) == 1:
            for i in root.val:
                try:
                    s=stateList[int(i)]
                    root.val = set([stateList[int(i)]])
                except Exception as e:
                    # 抓住出现(1)情况错误
                    root.val = set([stateList[int(i[1:-1])]])
                break

    # 后序遍历打印
    def postorderPrint(self, root):
        if (root is None):
            return
        self.postorderPrint(root.left)
        self.postorderPrint(root.right)
        print(root.val)

    def postorderPrintTracker(self, root):
        if (root is None):
            return
        self.postorderPrintTracker(root.left)
        self.postorderPrintTracker(root.right)
        print(root.tracker)

    def firstUppass(self, root,father):
        if root == None:return
        if len(root.val)>1:
            if father.val in root.val:
                root.val=father.val

        self.firstUppass(root.left,root)
        self.firstUppass(root.right,root)

    def firstDownpass(self, root):
        if root == None:return
        if root.left == None or root.right == None :
            return
        self.firstDownpass(root.right)
        self.firstDownpass(root.left)


        if  root.left.val & root.right.val :
            root.val=root.left.val & root.right.val
        else:
            root.val=root.left.val | root.right.val


    def initialiseTracker(self, root, fatherRoot):
        if root == None: return
        if fatherRoot!=None and root.val !=fatherRoot.val:
            root.tracker=True
        self.initialiseTracker(root.left, root)
        self.initialiseTracker(root.right, root)

    def secondDownPass(self, root, score):
        if root == None: return
        self.secondDownPass(root.left, score)
        self.secondDownPass(root.right, score)

        if root.right == None or root.left == None:
            return

        if root.tracker:
            score[0]+=1

def getFitchs(treeResult, characters):
    count = 0
    for i in range(1, len(characters[0])):
        count += getFitch(treeResult, characters[:, i])
    return count

def getFitch(treeResult, character):
    tree = Tree(treeResult)

    tree.postorderInit(tree.root, character)

    tree.postorderGetAllElement(tree.root)

    tree.firstUppass(tree.root, None)

    tree.firstDownpass(tree.root)







    tree.initialiseTracker(tree.root,None)

    tree.secondDownPass(tree.root, tree.score)

    # tree.secondUpPass(tree.root, False)
    return tree.score[0]

if __name__ == "__main__":
    path = r"testData\011号简化数据集奇虾\011号完整数据集.txt"
    data = readDataTxt(path)
    li = np.array(data)
    te = ['(((1),(3,4)),(2,((5,6),(7,((0),(8,9))))))', '(((1),(3,4)),(2,((5,6),((0,7),(8,9)))))', '(((1),(3,4)),(2,(6,(5,((0,7),(8,9))))))', '(((1),(3,4)),(2,(6,(5,(7,((0),(8,9)))))))']
    for i in te:
        print("{}:{}".format(i, getFitchs(i, li)))
