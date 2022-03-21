# 701 · Trim a Binary Search Tree

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# Method.1      DFS
class Solution:
    """
    @param root: given BST
    @param minimum: the lower limit
    @param maximum: the upper limit
    @return: the root of the new tree 
    """
    def trimBST(self, root, minimum, maximum):
        # write your code here
        if root is None:
            return root
        # 若根节点的值小于最小值，则递归调用右子树并返回右子树；
        if root.val < minimum :
            return self.trimBST(root.right, minimum, maximum)
        # 若根节点的值大于最大值，则递归调用左子树并返回左子树；
        elif root.val > maximum :
            return self.trimBST(root.left, minimum, maximum)
        # 否则修剪左子树，右子树并返回根节点。
        else:
            root.left = self.trimBST(root.left, minimum, maximum)       # maximum可以换成node.val
            root.right = self.trimBST(root.right, minimum, maximum)     # minimum可以换成node.val
        return root


# Method.2      DFS
class Solution:
    """
    @param root: given BST
    @param minimum: the lower limit
    @param maximum: the upper limit
    @return: the root of the new tree 
    """
    def trimBST(self, root, minimum, maximum):
        
        return self.dfs(root, minimum, maximum)
        
    def dfs(self, root, minimum, maximum):
        if root is None:
            return None

        root.left = self.dfs(root.left, minimum, maximum)
        root.right = self.dfs(root.right, minimum, maximum)
        
        if root.val <= maximum and root.val >= minimum:
            return root
        elif root.left and root.left.val <= maximum and root.left.val >= minimum:
            return root.left
        elif root.right and root.right.val <= maximum and root.right.val >= minimum:
            return root.right
        
        return None



# Method.3
"""  
因为涉及删除节点，凡是删除的问题，都会涉及到结构的变化，也就意味着 root 可能也会被删掉。
类似很多 LinkedList 的题目，我们给 root 增加一个伪父亲节点 dummy。然后将父亲节点和当前节点一起传递到递归函数中进行处理。
分别实现 trim_min 和 trim_max 来做两次不同的修剪，遍历 BST 两次，可解决问题。

时间复杂度 O(N)
额外空间复杂度没有（不含递归耗费）
"""
class Solution:
    """
    @param root: given BST
    @param minimum: the lower limit
    @param maximum: the upper limit
    @return: the root of the new tree 
    """
    def trimBST(self, root, minimum, maximum):
        dummy = TreeNode(float('inf'))
        dummy.left = root
        
        self.trim_min(dummy, dummy.left, minimum)
        self.trim_max(dummy, dummy.left, maximum)
        return dummy.left

    def trim_min(self, parent, node, minimum):
        if node is None:
            return
        if node.val >= minimum:
            self.trim_min(node, node.left, minimum)
            return
        if parent.left == node:
            parent.left = node.right
        else:
            parent.right = node.right
        self.trim_min(parent, node.right, minimum)
        
    def trim_max(self, parent, node, maximum):
        if node is None:
            return
        if node.val <= maximum:
            self.trim_max(node, node.right, maximum)
            return
        if parent.left == node:
            parent.left = node.left
        else:
            parent.right = node.left
        self.trim_max(parent, node.left, maximum)