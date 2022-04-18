# 649 · Binary Tree Upside Down
"""
Given a binary tree where all the right nodes are either leaf nodes with a sibling 
(a left node that shares the same parent node) or empty, flip it upside down and turn it 
into a tree where the original right nodes turned into left leaf nodes. Return the new root.
Input: {1,2,3,4,5}
Output: {4,5,2,#,#,3,1}
Explanation:
The input is
    1
   / \
  2   3
 / \
4   5
and the output is
   4
  / \
 5   2
    / \
   3   1
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the root of binary tree
    @return: new root
    """
    def upsideDownBinaryTree(self, root):
        if root is None:
            return None

        return self.dfs(root, None)
        
    def dfs(self, node, parent):
        if node is None:
            return parent

        newRoot = self.dfs(node.left, node)

        if parent:
            node.left = parent.right
            node.right = parent

            parent.left = None
            parent.right = None
        
        return newRoot