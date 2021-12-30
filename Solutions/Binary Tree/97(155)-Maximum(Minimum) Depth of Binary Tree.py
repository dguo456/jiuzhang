# 97 · Maximum Depth of Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
from collections import deque

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    # Divide and Conquer
    def maxDepth(self, root):
        if not root:
            return 0

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        # ***这里注意不用分别讨论左子树右子树是否存在的情况，对比Minimumm***
        return max(left_depth, right_depth) + 1

    # Traversal
    def maxDepth(self, root):
        if not root:
            return 0

        self.result = 1
        self.dfs(root, 1)
        return self.result

    def dfs(self, root, curr_depth):
        if root is None:
            return

        if root.left is None and root.right is None:
            self.result = max(self.result, curr_depth)

        if root.left:
            self.dfs(root.left, curr_depth + 1)
        if root.right:
            self.dfs(root.right, curr_depth + 1)


    # BFS
    def maxDepth(self, root):
        if root is None:
            return 0

        queue = deque([root])
        result = 1

        curr_depth = 0
        while queue:
            curr_depth += 1

            for i in range(len(queue)):
                node = queue.popleft()
                
                if node.left is None and node.right is None:
                    result = max(result, curr_depth)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return result



# 155 · Minimum Depth of Binary Tree
import sys

class Solution:
    """
    @param root: The root of binary tree
    @return: An integer
    """
    # Divide and Conquer
    def minDepth(self, root):
        if root is None:
            return 0

        left_sum = self.minDepth(root.left)
        right_sum = self.minDepth(root.right)

        if left_sum == 0:
            return right_sum + 1
        if right_sum == 0:
            return left_sum + 1
        return min(left_sum, right_sum) + 1

    # Traversal
    def minDepthII(self, root):
        if root is None:
            return 0

        self.result = sys.maxsize    # 取最小值要初始化成最大
        self.traversal(root, 1)
        return self.result

    def traversal(self, root, curr_depth):
        if root is None:
            return

        if root.left is None and root.right is None:
            self.result = min(self.result, curr_depth)

        if root.left:
            self.traversal(root.left, curr_depth + 1)
        if root.right:
            self.traversal(root.right, curr_depth + 1)

    # BFS
    def minDepthIII(self, root):
        if not root:
            return 0
        queue = deque([root])
        depth = 0

        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left is None and node.right is None:
                    return depth
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth