# 597 · Subtree with Maximum Average
"""
Given a binary tree, find the subtree with maximum average. Return the root of the subtree.

Input:
{1,-5,11,1,2,4,-2}
Output: 11
Explanation:
The tree is look like this:
     1
   /   \
 -5     11
 / \   /  \
1   2 4    -2 
The average of subtree of 11 is 4.3333, is the maximun.
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """
    # 这里定义两个全局变量
    average, node = 0, None
    
    def findSubtree2(self, root):
        
        self.helper(root)
        return self.node
        
    def helper(self, root):
        if root is None:
            return 0, 0
            
        left_sum, left_size = self.helper(root.left)
        right_sum, right_size = self.helper(root.right)
        
        sums, size = left_sum + right_sum + root.val, left_size + right_size + 1
        
        if self.node is None or sums * 1.0 / size > self.average:
            self.node = root
            self.average = sums * 1.0 / size
            
        return sums, size