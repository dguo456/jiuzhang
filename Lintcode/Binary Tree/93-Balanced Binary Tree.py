# 93 · Balanced Binary Tree
"""
Given a binary tree, determine if it is height-balanced.
For this problem, a height-balanced binary tree is defined as 
a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

需要注意Result Type的用法
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        if not root:
            return True

        balanced, _ = self.dfs(root)
        return balanced

    def dfs(self, root):
        if root is None:
            return True, 0

        balanced, left_height = self.dfs(root.left)
        if not balanced:
            return False, left_height  # 第二个任何值都可以，0也可以

        balanced, right_height = self.dfs(root.right)
        if not balanced:
            return False, right_height  # 第二个任何值都可以，0也可以

        return abs(left_height - right_height) <= 1, max(left_height, right_height) + 1



# Method.2  在树上做一次DFS，计算以每个点为根的子树高度。
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        is_balanced, _ = self.helper(root)
        return is_balanced
        
    def helper(self, root):
        if not root:
            return True, 0
        
        is_left_balanced, left_height = self.helper(root.left)
        is_right_balanced, right_height = self.helper(root.right)
        
        root_height = max(left_height, right_height) + 1
        
        if not is_left_balanced or not is_right_balanced:
            return False, root_height
            
        if abs(left_height - right_height) > 1:
            return False, root_height
            
        return True, root_height