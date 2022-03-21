# Method.1  recursion
"""
考点: 搜索遍历
题解: 首先判断两树根节点开始是否相同，如果不相同，就从T1的子树和T2当前节点递归搜索。
"""

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param T1: The roots of binary tree T1.
    @param T2: The roots of binary tree T2.
    @return: True if T2 is a subtree of T1, or false.
    """
    def isSubtree(self, T1, T2):
        temp = []
        self.traversal(T1, temp)
        t1 = ','.join(temp)

        temp = []
        self.traversal(T2, temp)
        t2 = ','.join(temp)

        return t1.find(t2) != -1

    def traversal(self, root, temp):
        if root is None:
            temp.append('#')
            return

        temp.append(str(root.val))
        self.traversal(root.left, temp)
        self.traversal(root.right, temp)


# Method.2 
"""
据题意，T1 可以有上百万个节点，所以不用递归遍历，免得面试官诘问你栈溢出怎么办。
定义一个判断俩个树相等的子函数，而不是子树，更方便。
"""
class Solution:
    
    def isSubtree(self, T1, T2):
        
        if not T2: return True 
        if not T1: return False
        
        stack = [T1]
        while stack:
            root = stack.pop()
            if self.isSametree(root, T2):
                return True
            if root.right:
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
        
        return False
        
    def isSametree(self, T1, T2):
        if not T1 and not T2: return True 
        if not (T1 and T2): return False 
        
        if T1.val != T2.val: return False 
        
        return self.isSametree(T1.left, T2.left) and self.isSametree(T1.right, T2.right)