# 262 · heir tree
"""
Please design a data structure MyTreeNodeof the heir tree, which contains the following methods:

addNode(MyTreeNode, val): Add a heir numbered valval to a person.
deleteNode(MyTreeNode): Disqualifies a person from inheritance (but does not affect the inheritance of his heirs)
traverse(MyTreeNode): Query the inheritance order under a person. 
    The inheritance order is traversed in the preorder traversal, which is, for each node, 
    first it itself, and then it traverses each of its heirs in the order from first to last.
At the same time, you need to complete a constructor of the class that only receives valval, 
and we will use it to construct a node with val = 1val=1 as the root of the heir tree.
"""

"""
考察点
树的数据结构的设计和先序遍历

题目分析
题目要求设计一颗继承人树，并可以先序遍历。我们发现这是一颗普通的nn叉树。因此，我们只需要按照nn叉树的数据结构设计即可。
删除节点的时候，为了保证后续节点的可遍历性。我们可以直接把对应的节点增加删除标记，当进行遍历的时候，我们不把它放入结果汇总。

复杂度
插入操作: 时间复杂度: O(1)
删除操作: 时间复杂度: O(1)
遍历操作: 时间复杂度: O(n)，其中n是插入过的节点数。
"""

class MyTreeNode:
    """
    @param val: the node's val
    @return: a MyTreeNode object
    """
    def __init__(self, val):
        self.val = val
        self.child = []
        self.parent = None
        self.is_deleted = False

    def addNode(self, root, num):
    	# root是父亲，root生了个孩子num,一定加在child列表的最后
    	# 输入一个treeNode
        newNode = MyTreeNode(num)
        newNode.parent = root
        root.child.append(newNode)
        return newNode

    def deleteNode(self, root):
        root.is_deleted = True
        return

    def traverse(self, root):
        path = []
        self.traverse_helper(root, path)
        return path

    def traverse_helper(self, root, path):
        if not root.is_deleted:
            path.append(root.val)
        for child in root.child:
            self.traverse_helper(child, path)
