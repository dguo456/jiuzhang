# 480 · Binary Tree Paths
"""
Given a binary tree, return all root-to-leaf paths.
Input: {1,2,3,#,5}
Output: ["1->2->5","1->3"]
"""

# Method.1   Traversal

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        if root is None:
            return []
        
        result = []
        self.traverse(root, [str(root.val)], result)
        return result
        
    def traverse(self, node, path, result):
        
        if node.left is None and node.right is None:
            result.append('->'.join(path))
            return
        
        if node.left:
            path.append(str(node.left.val))
            self.traverse(node.left, path, result)
            path.pop()  # 回溯 back
            
        if node.right:
            path.append(str(node.right.val))
            self.traverse(node.right, path, result)
            path.pop()  



# Method.2      Divide&Conquer
class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        if root is None:
            return []
            
        # 99% 的题，不用单独处理叶子节点
        # 这里需要单独处理的原因是 root 是 None 的结果，没有办法用于构造 root 是叶子的结果
        if root.left is None and root.right is None:
            return [str(root.val)]

        leftPaths = self.binaryTreePaths(root.left)
        rightPaths = self.binaryTreePaths(root.right)
        
        paths = []
        for path in leftPaths + rightPaths:
            paths.append(str(root.val) + '->' + path)
            
        return paths



# Method.3    Non-Recursive
class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        if root is None:
            return []
        res = []
        queue = [(root, '')]
        
        while len(queue) > 0:
            current, path = queue.pop(0)
            if path:
                path += '->' + str(current.val)
            else:
                path += str(current.val)
                
            if not current.left and not current.right:  # 判断是否为叶节点，如果是，将路径加入 res
                res.append(path)
            if current.left:
                queue.append((current.left, path))
            if current.right:
                queue.append((current.right, path))
        
        return res