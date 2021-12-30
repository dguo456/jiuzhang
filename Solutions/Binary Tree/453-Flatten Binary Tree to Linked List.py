# 453 · Flatten Binary Tree to Linked List
"""
遍历
遍历就是沿着某条搜索路线依次对树中每个结点均做一次且仅做一次访问，在访问的过程中就利用全局变量纪录结果，
遍历没返回值, 遍历要么定义全局变量记录结果；要么在主函数中定义，每次作为形参传递给辅助函数

分治
分治是将一个问题分成多个子问题，子问题的解可以合并为该问题的完整解， 合起来这一步是关键，分治需要返回值

递归只是一种实现方式，分治和递归结合起来比较方便，遍历也可以用非递归实现

分治＋遍历的算法可以用分治加ResultType实现
需要多个返回值时定义一个ResultType类型
一般的题都既可以用遍历做也可以用分治来做
"""

# Method.1    标准分治
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        self.flatten_and_return_last_node(root)

    def flatten_and_return_last_node(self, root):
        if root is None:
            return None

        left_last = self.flatten_and_return_last_node(root.left)
        right_last = self.flatten_and_return_last_node(root.right)
        
        if left_last:
            left_last.right = root.right
            root.right = root.left
            root.left = None

        # return  right_last or left_last or root
        if right_last:
            return right_last
        if left_last:
            return left_last
        if root:
            return root



# Method.2    优化版，只用一个变量last_node来记录，相当于left_last,省去了right_last,但是要每次存当前层的root.right
class SolutionII:
    last_node = None
    
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        if root is None:
            return
        
        if self.last_node:
            self.last_node.left = None
            self.last_node.right = root
            
        self.last_node = root
        right = root.right
        self.flatten(root.left)
        self.flatten(right)



# Method.3    Non-Recursive
"""
利用stack先序遍历 （根->右->左，入栈）
连接操作只需要把每个 pop 出来的node:
1.左子树赋为None
2.右子树赋为当前栈顶（如果栈顶不空），否则赋为None
"""

from collections import deque

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        """Do not return anything, modify root in-place instead."""
        
        if not root:
            return
        
        stack = deque([root])
        
        while stack:
            node = stack.pop()
            
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            
            node.left = None
            if stack:
                node.right = stack[-1]
            else:
                node.right = None