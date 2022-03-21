# 921 · Count Univalue Subtrees
"""
Given a binary tree, count the number of uni-value subtrees.
A Uni-value subtree means all nodes of the subtree have the same value.

Input:  root = {5,1,5,5,5,#,5}
Output: 4
Explanation:
              5
             / \
            1   5
           / \   \
          5   5   5
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# Method.1
class Solution:
    """
    @param root: the given tree
    @return: the number of uni-value subtrees.
    """
    def countUnivalSubtrees(self, root):
        if root is None:
            return 0

        self.result = 0
        self.helper(root)

        return self.result

    def helper(self, root):
        if root is None:
            return True

        left = self.helper(root.left)
        right = self.helper(root.right)

        if left and right and (not root.left or root.left.val == root.val) and \
            (not root.right or root.right.val == root.val):
            self.result += 1
            return True

        return False



# Method.2
class Solution:
    """
    @param root: the given tree
    @return: the number of uni-value subtrees.
    """
    def countUnivalSubtrees(self, root):
        if root is None:
            return 0

        self.count = 0
        self.helper(root)

        return self.count

    def helper(self, root):
        if root is None:
            return True, None

        left_exists, left_val = self.helper(root.left)
        right_exists, right_val = self.helper(root.right)

        if left_exists and right_exists and left_val is None and right_val is None:
            self.count += 1
            return True, root.val

        if left_exists and right_exists and left_val and right_val and left_val == root.val and right_val == root.val:
            self.count += 1
            return True, root.val

        if left_exists and right_exists and left_val and left_val == root.val:
            self.count += 1
            return True, root.val

        if left_exists and right_exists and right_val and right_val == root.val:
            self.count += 1
            return True, root.val

        return False, None