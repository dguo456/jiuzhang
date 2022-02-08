
# 11 · Search Range in Binary Search Tree
"""
这题考查的是二叉查找树的性质，以及二叉树的中序遍历。
二叉查找树满足左子树所有节点的值都小于当前节点的值，右子树所有节点的值都大于当前节点的值。二叉查找树的中序遍历是一个排好序的序列。
这题我们在中序遍历的过程中将在数值范围内的值按序加入到数组中，就能得到最终的结果。

代码思路
二叉树中序遍历
这题有一个可以剪枝的技巧，如果已经可以确定左子树或右子树不在数值范围内，可以不遍历相应的子树。

复杂度分析
设二叉树的节点数为N。

时间复杂度
遍历一遍二叉树的时间复杂度为O(N)。
空间复杂度
递归的空间开销取决于树的最大深度，空间复杂度为O(N)。
"""

# Method.1      标准DFS

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: param root: The root of the binary search tree
    @param k1: An integer
    @param k2: An integer
    @return: return: Return all keys that k1<=key<=k2 in ascending order
    """
    def searchRange(self, root, k1, k2):
        result = []
        self.travel(root, k1, k2, result)
        return result
    
    def travel(self, root, k1, k2, result):
        if root is None:
            return
    	# 剪枝，如果当前节点小于等于k1，不必访问左子树
        if root.val > k1:
            self.travel(root.left, k1, k2, result)
        if k1 <= root.val and root.val <= k2:
            result.append(root.val)
        # 剪枝，如果当前节点大于等于k2，不必访问右子树
        if root.val < k2:
            self.travel(root.right, k1, k2, result)



# Method.2      BFS
import sys

class Solution:
    """
    @param root: param root: The root of the binary search tree
    @param k1: An integer
    @param k2: An integer
    @return: return: Return all keys that k1<=key<=k2 in ascending order
    """
    def searchRange(self, root, k1, k2):
        if not root:
            return []

        dummy = TreeNode(-sys.maxsize)
        dummy.right = root
        stack = [dummy]
        results = []

        while stack:
            node = stack.pop()
            if node.val > k2:
                break
            if k1 <= node.val:
                results.append(node.val)
            if node.right:
                node = node.right
                while node:
                    stack.append(node)
                    node = node.left

        return results

# Another Method of BFS
class Solution:
    """
    @param root: The root of the binary search tree.
    @param k1 and k2: range k1 to k2.
    @return: Return all keys that k1<=key<=k2 in increasing order.
    """     
    def searchRange(self, root, k1, k2):
        if root is None:
            return []

        results = []
        queue = [root]
        index = 0
        
        while index < len(queue):
            if queue[index] is not None:
                if queue[index].val >= k1 and queue[index].val <= k2:
                    results.append(queue[index].val)

                queue.append(queue[index].left)
                queue.append(queue[index].right)

            index += 1

        return sorted(results)