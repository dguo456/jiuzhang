# Method.1 分治
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        if root is None:
            return

        left = self.invertBinaryTree(root.left)
        right = self.invertBinaryTree(root.right)

        root.left, root.right = right, left
        return root

# Method. 2  递归
class Solution:
    # @param root: a TreeNode, the root of the binary tree
    # @return: nothing
    class Solution:
        """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        self.dfs(root)
        return root

    def dfs(self, root):
        if root is None:
            return

        root.left, root.right = root.right, root.left
        if root.left:
            self.dfs(root.left)
        if root.right:
            self.dfs(root.right)


# Method.3 非递归
import collections

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        if not root:
            return
        
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            node.left, node.right = node.right, node.left
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
