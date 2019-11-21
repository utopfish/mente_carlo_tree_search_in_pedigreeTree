# @Time:2019/11/13 15:48
# @Author:liuAmon
# @e-mail:utopfish@163.com
# @File:fitch.py
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


if __name__ == "__main__":
    treeResult = "((0,1),(2,(3,4)))"
    tree = Tree(treeResult)
