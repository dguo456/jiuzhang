# 864 · Equal Tree Partition
"""
Given a binary tree with n nodes, your task is to check if it's possible to partition the tree 
to two trees which have the equal sum of values after removing exactly one edge on the original tree.
"""

# Input: {5,10,10,#,#,2,3}
# Output: true
# Explanation:
#   origin:
#      5
#     / \
#    10 10
#      /  \
#     2    3
#   two subtrees:
#      5       10
#     /       /  \
#    10      2    3


# Method.1      DFS
"""
从一个节点遍历它的子树就可以得到以它为根节点的子树的节点值总和.
我们可以先从根节点进行一次遍历, 然后就可以得到整棵树的节点值总和 sum
然后再进行一次遍历, 在这一次遍历的过程中判断每个节点的子树节点值总和是否等于 sum / 2, 如果有返回true即可.
也可以借助 set (或 map) 只进行一次遍历, 即第一次遍历时把所有的权值总和放到 set 里, 最后获得整棵树的节点值总和时, 
直接判断集合里是否有 sum / 2 即可.
"""
# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: a TreeNode
    @return: return a boolean
    """
    def checkEqualTree(self, root):
        self.map_dict = {}
        tree_sum = self.dfs(root)
        if tree_sum == 0:
            return self.map_dict[0] > 1
        return tree_sum % 2 == 0 and not self.map_dict.get(tree_sum / 2) == None

    def dfs(self, root):
        if root is None:
            return 0
        tree_sum = root.val + self.dfs(root.left) + self.dfs(root.right)
        if(self.map_dict.get(tree_sum) == None):
            self.map_dict[tree_sum] = 1
        else:
            self.map_dict[tree_sum] += 1
        return tree_sum



# Method.2      Memorization + BFS
"""
Time:O(N), Space:O(N)
先計算一次total sum, 順便把每個node的數值記起來(Memorization)
接著用BFS遍歷整棵樹
當目前node往下數值的sum == 整棵樹的sum - 目前node往下數值的sum時
表示可以一刀分開, return True
做完BFS沒找到return False
"""
from collections import deque

class Solution:
    """
    @param root: a TreeNode
    @return: return a boolean
    """
    def checkEqualTree(self, root):
        if not root:
            return False

        self.memo = {}
        total_sum = self.traversal(root)
        # 如果是奇数则直接返回False
        if total_sum % 2 == 1:
            return False

        # BFS
        queue = deque([root])
        while queue:
            node = queue.popleft()
            curr_sum = self.traversal(node)

            if node != root and curr_sum == total_sum - curr_sum:
                return True
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return False

    def traversal(self, root):
        if root in self.memo:
            return self.memo[root]
        if not root:
            return 0
        if not root.left and not root.right:
            self.memo[root] = root.val
            return self.memo[root]

        left = self.traversal(root.left)
        right = self.traversal(root.right)

        self.memo[root] = root.val
        if left:
            self.memo[root] += left
        if right:
            self.memo[root] += right
            
        return self.memo[root]