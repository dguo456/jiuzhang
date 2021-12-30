# 94 · Binary Tree Maximum Path Sum

"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
import sys

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxPathSum(self, root):
        self.maxSum = -sys.maxsize-1 
        self.dfs(root)
        return self.maxSum

    def dfs(self, root):
        if root is None:
            return 0
        
        leftSum = self.dfs(root.left)
        rightSum = self.dfs(root.right)

        # 在遍历树的过程中，要看两样，一是当前subtree子树的最大值，一是当前subtree能对parent tree父树贡献什么。
        self.maxSum = max(self.maxSum, root.val + leftSum + rightSum)
        return max(leftSum + root.val, rightSum + root.val, 0)



# 475 · Binary Tree Maximum Path Sum II
class Solution:
    """
    @param root: the root of binary tree.
    @return: An integer
    """
    def maxPathSum2(self, root):
        self.result = -sys.maxsize
        value_sum = 0
        self.dfs(root, value_sum)
        return self.result
        
    def dfs(self, root, value_sum):
        value_sum += root.val
        
        if value_sum > self.result:
            self.result = value_sum 
        
        if root.left is not None:
            self.dfs(root.left, value_sum)
        
        if root.right is not None:
            self.dfs(root.right, value_sum)