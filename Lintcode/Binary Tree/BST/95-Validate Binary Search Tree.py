# 95 · Validate Binary Search Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        
        self.last_val = None
        self.is_BST = True
        self.dfs(root)
        return self.is_BST
        
    def dfs(self, root):
        
        if root is None:
            return
        
        self.dfs(root.left)
        # first need to check if self.last_val exist or not
        if self.last_val and self.last_val >= root.val:
            self.is_BST = False
            return
        # ***每次都需要把前一个值更新成当前值***
        self.last_val = root.val
        self.dfs(root.right)