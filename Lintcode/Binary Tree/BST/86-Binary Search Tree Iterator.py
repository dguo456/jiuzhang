# 86 · Binary Search Tree Iterator
"""
这是一个非常通用的利用 stack 进行 Binary Tree Iterator 的写法。

stack 中保存一路走到当前节点的所有节点，stack.peek() 一直指向 iterator 指向的当前节点。
因此判断有没有下一个，只需要判断 stack 是否为空
获得下一个值，只需要返回 stack.peek() 的值，并将 stack 进行相应的变化，挪到下一个点。

挪到下一个点的算法如下:
1.  如果当前点存在右子树，那么就是右子树中“一路向西”最左边的那个点
2.  如果当前点不存在右子树，则是走到当前点的路径中，第一个左拐的点
访问所有节点用时O(n)，所以均摊下来访问每个节点的时间复杂度时O(1)
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
Example of iterate a tree:
iterator = BSTIterator(root)
while iterator.hasNext():
    node = iterator.next()
    do something for node 
"""
class BSTIterator:
    #@param root: The root of binary tree.
    def __init__(self, root):
        self.stack = []
        self.curt = root

    #@return: True if there has next node, or false
    def hasNext(self):
        return self.curt is not None or len(self.stack) > 0

    #@return: return next node
    def _next(self):
        while self.curt is not None:
            self.stack.append(self.curt)
            self.curt = self.curt.left
            
        self.curt = self.stack.pop()
        nxt = self.curt
        self.curt = self.curt.right
        return nxt



# Method.2      更简洁的代码
class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        dummy = TreeNode(0)
        dummy.right = root
        self.stack = [dummy]
        self.next()

    """
    @return: True if there has next node, or false
    """
    def hasNext(self):
        # 这里不能写return self.stack不空，
        return bool(self.stack)

    """
    @return: return next node
    """
    def _next(self):
        node = self.stack.pop()
        next_node = node
        if node.right:
            node = node.right
            while node:
                self.stack.append(node)
                node = node.left
        return next_node