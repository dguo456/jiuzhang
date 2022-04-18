# 650 · Find Leaves of Binary Tree
"""
Given a binary tree, collect a tree's nodes as if you were doing this: 
Collect and remove all leaves, repeat until the tree is empty.
"""
# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param: root: the root of binary tree
    @return: collect and remove all leaves
    """
    # 对于叶子结点将其高度置为0。对于任一非叶子节点，其高度等于左右子树中高度的最大值加一。
    def findLeaves(self, root):
        if not root:
            return []

        self.results = []
        self.dfs(root)

        return self.results

    def dfs(self, root):
        if not root:
            return -1

        height = -1
        left_height = self.dfs(root.left)
        right_height = self.dfs(root.right)

        height = max(left_height, right_height) + 1

        if height >= len(self.results):
            self.results.append([])

        self.results[height].append(root.val)

        return height