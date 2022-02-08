# 902 · Kth Smallest Element in a BST
"""
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.
"""
# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# Method.1
"""
构建了一个dfs函数做inorder traversal，当访问到第k个时取值。
因为python对于原始数据类型不支持引用，所以用类的内部变量作为全局变量处理。
"""
class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        if not root:
            return None

        self.Kth, self.result = k, None
        self.dfs(root)
        return self.result

    def dfs(self, root):
        if root is None:
            return

        self.dfs(root.left)
        self.Kth -= 1
        if self.Kth == 0:
            self.result = root.val
            return
        self.dfs(root.right)



# Method.2      使用 Binary Tree Iterator，连续找 k 个点。
class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        # use binary tree iterator
        dummy = TreeNode(0)
        dummy.right = root
        stack = [dummy]
            
        for i in range(k):
            node = stack.pop()
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left
            if not stack:
                return None
                
        return stack[-1].val