import time
import numpy as np
import pandas as pd


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
                    root.val = set([stateList[int(i)]])
                except:
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

    def firstUppass(self, root, fatherRoot):
        if root == None:
            return
        if root.left == None and root.right == None and root.visited:
            return
        if self.hasInapp(root.val):
            # step2
            if self.hasApp(root.val):
                # step3
                tempFather = fatherRoot

                if (tempFather != False):
                    if self.inSet('-', tempFather.val):
                        root.val = set(['-'])
                        root.visited = True
                    else:
                        root.val.discard('-')
                        root.visited = True
                else:
                    root.val.discard('-')
                    root.visited = True
                self.firstUpass_step8(root)
            else:
                # step4
                tempFather = fatherRoot
                if (tempFather != False):
                    if self.inSet('-', tempFather.val):
                        root.val = set(['-'])
                        root.visited = True
                        self.firstUpass_step8(root)
                    else:
                        # step5
                        if self.hasApp(root.left.val) or self.hasApp(root.right.val) or self.inSet('?',
                                                                                                   root.left.val) or self.inSet(
                                '?', root.right.val):

                            root.val = self.union(root.left.val, root.right.val)

                            root.val.discard('-')
                            root.visited = True
                        else:
                            root.val = set(['-'])
                            root.visited = True
                        self.firstUpass_step8(root)
                else:
                    # step5
                    if self.hasApp(root.left.val) or self.hasApp(root.right.val):
                        root.val = self.union(root.left.val, root.right.val)

                        root.val.discard('-')
                        root.visited = True
                    else:
                        root.val = set(['-'])
                        root.visited = True
                    self.firstUpass_step8(root)
        else:
            # step8
            self.firstUpass_step8(root)

    def firstUpass_step8(self, root):
        if (root.left != None and root.right != None):
            if (root.left.left == None and root.left.visited == False) or (
                    root.right.right == None and root.right.visited == False):
                # step6
                if root.left.left == None and root.left.visited == False:
                    root.left.visited = True
                    if self.hasInapp(root.left.val) and self.hasApp(root.left.val):
                        # step7
                        if self.hasInapp(root.val) and self.hasApp(root.val) == False:
                            root.left.val = set(['-'])
                        else:
                            root.left.val.discard('-')
                        self.firstUpass_step8(root)
                    else:
                        self.firstUpass_step8(root)

                if root.right.right == None and root.right.visited == False:
                    root.right.visited = True
                    if self.hasInapp(root.right.val) and self.hasApp(root.right.val):
                        # step7
                        if self.hasInapp(root.val) and self.hasApp(root.val) == False:
                            root.right.val = set(['-'])
                        else:
                            root.right.val.discard('-')
                        self.firstUpass_step8(root)
                    else:
                        self.firstUpass_step8(root)
        self.firstUppass(root.left, root)
        self.firstUppass(root.right, root)

    def firstDownpass(self, root):
        if root.right == None or root.left == None:
            return
        self.firstDownpass(root.left)
        self.firstDownpass(root.right)
        temp = self.intersection(root.left.val, root.right.val)

        if (len(temp) > 0):
            # step2
            if len(temp) == 1 and self.inSet('-', temp) and self.hasApp(root.left.val) and self.hasApp(root.right.val):

                root.val = self.union(root.left.val, root.right.val)
            else:

                root.val = self.intersection(root.left.val, root.right.val)
        else:
            # step3
            if self.hasApp(root.left.val) and self.hasApp(root.right.val):

                root.val = self.union(root.left.val, root.right.val)
                root.val.discard('-')
            else:

                root.val = self.union(root.left.val, root.right.val)

        # 对缺失的处理
        if self.inSet('?', root.val):
            root.val.discard('?')

    def initialiseTracker(self, root, fatherRoot):
        if root == None: return
        if root.left == None:
            if self.hasInapp(root.val):
                root.tracker = False
            elif self.hasApp(root.val):
                root.tracker = True
            else:
                if self.hasInapp(fatherRoot.val):
                    root.tracker = False
                elif self.hasApp(fatherRoot.val):
                    root.tracker = True
        self.initialiseTracker(root.left, root)
        self.initialiseTracker(root.right, root)

    def secondDownPass(self, root, score):
        if root == None: return
        self.secondDownPass(root.left, score)
        self.secondDownPass(root.right, score)

        if root.right == None or root.left == None:
            return

        if root.left.tracker or root.right.tracker:
            root.tracker = True
        else:
            root.tracker = False
        if self.hasApp(root.val):
            # step3
            # 对？的处理

            temp = self.intersection(root.left.val, root.right.val)

            if len(temp) > 0:
                # step4
                if self.hasApp(temp) and self.hasInapp(temp) == False:
                    root.val = temp
                else:
                    root.val = set(['-'])
            else:
                # step5
                temp = self.union(root.left.val, root.right.val)

                temp.discard("-")
                root.val = temp
                # step6
                if self.hasApp(root.left.val) and self.hasApp(root.right.val):
                    score[0] += 1
                else:
                    # step7
                    if root.right.tracker and root.left.tracker:
                        score[0] += 1
                    else:
                        # step8
                        if root.right.tracker and root.left.tracker and self.hasInapp(root.val) and len(root.val) == 1:
                            score[0] += 1
                    return
                return
        else:
            # step7
            if root.right.tracker and root.left.tracker:
                score[0] += 1
            else:
                # step8
                if root.right.tracker and root.left.tracker and self.hasInapp(root.val) and len(root.val) == 1:
                    score[0] += 1
            return

    def secondUpPass(self, root, fatherRoot):
        if root == None:
            return
        if self.hasApp(root.val):
            # step2
            if fatherRoot != False and self.hasApp(fatherRoot.val):
                # step3
                if self.isEqual(fatherRoot.val, root.val):
                    pass
                else:
                    # step4
                    if root.left == None or root.right == None:
                        return
                    temp = self.intersection(root.left.val, root.right.val)
                    if len(temp) > 0:
                        # step5
                        temp = self.union(root.left.val, root.right.val)
                        for i in self.intersection(fatherRoot.val, temp):
                            root.val.add(i)
                        pass
                    else:
                        # step6
                        if self.hasInapp(root.left.val) and self.hasInapp(root.right.val):
                            # step7
                            temp = self.intersection(root.left.val, root.right.val)
                            if len(self.intersection(fatherRoot.val, temp)) > 0:
                                root.val = fatherRoot.val
                            else:
                                # 此步存疑，看将来的理解
                                root.val = set(self.allElement)
                                root.val.discard("-")
                                root.val.discard("?")
                            pass
                        else:
                            # step8
                            for i in fatherRoot.val:
                                root.val.add(i)
        self.secondUpPass(root.left, root)
        self.secondUpPass(root.right, root)

    def hasInapp(self, node_value):
        for i in node_value:
            if i == '-':
                return True
        return False

    def hasApp(self, node_value):
        for i in node_value:
            if i != '-' and i != '?':
                return True
        return False

    def inSet(self, set_value, set_list):
        for i in set_list:
            if set_value == i:
                return True
        return False

    def union(self, set1, set2):
        if "?" in set1 or "?" in set2:
            temp = set(self.allElement)
            temp.discard("?")
            return temp
        else:
            return set1.union(set2)

    def intersection(self, set1, set2):
        if "?" in set1 or "?" in set2:
            temp = set1.union(set2)
            temp.discard("?")
            return temp
        else:
            return set1.intersection(set2)

    def isEqual(self, set_list1, set_list2):
        if len(set_list1) != len(set_list2):
            return False
        for i in set_list1:
            if i not in set_list2:
                return False
        return True

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


def readDataTxt(path):
    data = pd.read_table(path, header=None, sep=" ")
    return data


def getSingleChararcterFitch(treeResult, character):
    tree = Tree(treeResult)

    tree.postorderInit(tree.root, character)

    tree.postorderGetAllElement(tree.root)

    tree.firstDownpass(tree.root)

    tree.firstUppass(tree.root, False)

    tree.initialiseTracker(tree.root, False)

    tree.secondDownPass(tree.root, tree.score)

    # tree.secondUpPass(tree.root, False)
    return tree.score[0]


def getsingleFitchs(treeResult, characters):
    count = 0
    for i in range(1, len(characters[0])):
        count += getSingleChararcterFitch(treeResult, characters[:, i])
    return count


def getFict2(treeResult, characters):
    star = time.time()
    count = 0

    for i in range(1, len(characters[0])):
        tree = Tree(treeResult)
        tree.postorderInit(tree.root, characters[:, i])
        tree.postorderGetAllElement(tree.root)
        tree.firstDownpass(tree.root)
        tree.firstUppass(tree.root, False)
        tree.initialiseTracker(tree.root, False)
        tree.secondDownPass(tree.root, tree.score)
        count += tree.score[0]
    print("fitct 耗时{}".format(time.time() - star))
    return count


if __name__ == "__main__":

    path = r"testData\011号简化数据集奇虾\011号完整数据集.txt"
    data = readDataTxt(path)
    li = np.array(data)

    te = ["((0,1),(2,(3,4)))", "({0,1,2,3},4)"]
    for i in te:
        print("{}:{}".format(i, getsingleFitchs(i, li)))
