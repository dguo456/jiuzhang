# 1165 · Subtree of Another Tree
"""
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure 
and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all 
of this node's descendants. The tree s could also be considered as a subtree of itself.

Given tree s:
     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param s: the s' root
    @param t: the t' root
    @return: whether tree t has exactly the same structure and node values with a subtree of s
    """
    def isSubtree(self, s, t):
        if s is None:
            return t is None
        if s.val == t.val and self.compare(s, t):
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
        
    def compare(self, s, t):
        if s is None:
            return t is None
        if t is None or s.val != t.val:
            return False
            
        return self.compare(s.left, t.left) and self.compare(s.right, t.right)