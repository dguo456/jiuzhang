# 163 · Unique Binary Search Trees
"""
Given n, how many structurally unique BSTs (binary search trees) that store values 1...n?

Input:n = 3,
Output: 5
Explanation:there are a total of 5 unique BST's.
"""
class Solution:
    """
    @param n: An integer
    @return: An integer
    """
    def numTrees(self, n):
        dp = [1, 1, 2]
        if n <= 2:
            return dp[n]
        else:
            dp += [0 for i in range(n-2)]
            for i in range(3, n+1):
                for j in range(1, i+1):
                    dp[i] += dp[j-1] * dp[i-j]
        return dp[n]



# 164 · Unique Binary Search Trees II
"""
Given n, generate all structurally unique BST's (binary search trees) that store values 1...n.
算法 : 二叉查找树

解题思路
这题需要构造出所有不同的二叉查找树。
由于二叉查找树的性质，对于一个二叉查找树的节点，它的左子树中所有节点的值都小于它的值，
它的右子树中所有节点的值都大于它的值。而且二叉查找树的中序遍历是一个排好序的数组。
我们可以枚举根节点的所有可能的值。如果根节点的值等于i，那么左子树的节点值在范围[1,i-1]内，右子树节点的值在范围[i+1,n]中。
遍历左子树的所有情况和右子树的所有情况，就可以组合出当根节点的值等于i时的所有情况。
根据以上的推导，我们可以发现这是一个明显的递归结构，左右子树也可以用这个递归结构构造。那么我们就可以用递归来解决这个问题。

代码思路
递归的步骤:
1、递归出口，如果start值大于end值，返回一个只含空节点的列表。
2、枚举根节点的值root_val，从start到end。
3、递归获得所有可能的左子树[start, root_val - 1]。
4、递归获得所有可能的右子树[root_val + 1, end]。
5、遍历左右子树的所有可能，组合成新的树，加入结果数组result中。
6、返回结果数组result。

复杂度分析
设二叉查找树的节点数为N。
C(n)=(2n)!/(n!(n+1)!)为卡特兰数第n项的计算公式。
所有不同的节点数为NN的二叉查找树为对应的卡特兰数C(N)。

时间复杂度: O(N*C(N))
由于有的子树会被重复计算，时间复杂度为O(N*C(N))。

空间复杂度: O(N)
空间复杂度取决于树的最大深度，O(N)。
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    # @paramn n: An integer
    # @return: A list of root
    def generateTrees(self, n):
        return self.dfs(1, n)
    
    # 返回值是从start到end的所有可能的二叉查找树
    def dfs(self, start, end):
        if start > end:
            return [None]
        
        result = []
        # 枚举根节点的值
        for root_val in range(start, end + 1):
            # 递归获得所有可能的左子树
            left_trees = self.dfs(start, root_val - 1)
            # 递归获得所有可能的右子树
            right_trees = self.dfs(root_val + 1, end)
            # 枚举每一种左右子树的组合，组成新的二叉树
            for left_tree in left_trees:
                for right_tree in right_trees:
                    root = TreeNode(root_val)
                    root.left = left_tree
                    root.right = right_tree
                    result.append(root)
        
        return result