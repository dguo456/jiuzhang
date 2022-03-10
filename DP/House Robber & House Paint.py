# 392 · House Robber
# 534 · House Robber II
# 535 · House Robber III
################################################################################################################



# 392 · House Robber
"""
You are a professional robber planning to rob houses along a street. Each house has a certain amount of 
money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have 
security system connected and it will automatically contact the police if two adjacent houses were broken into 
on the same night. Given a list of non-negative integers representing the amount of money of each house, 
determine the maximum amount of money you can rob tonight without alerting the police.
"""
# 解题思路
# 由抢房屋的性质可以看出，抢前i个房屋能得到的最大值，与后面如何抢的方案无关，只与前i - 1个房屋的最优方案有关。
# 这满足了动态规划的无后效性和最优子结构。同时，由于题目不能抢相邻房屋，那么如果抢了第i个房屋，就不能抢第i - 1个房屋，
# 可以得出前i个的最优方案也与前i - 2个的最优方案有关。
# 代码思路
# 可以设dp(i, 0)为如果不抢第i个房屋，前i个房屋的最优方案为多少；设dp(i, 1)为如果抢第i个房屋，前i个房屋的最优方案为多少。
# 可以得出以下的状态转移方程式：
# dp(i, 0) = max(dp(i−1, 0), dp(i−1, 1))
# dp(i, 1) = A[i] + dp(i-1, 0)

# 复杂度分析    设房屋数量为N。
# 时间复杂度:
# 扫描一遍数组，每次O(1)时间动态转移，复杂度为O(N)。
# 空间复杂度:
# dp数组的规模为N * 2，空间复杂度为O(N)。
# 在空间上可以用滚动数组进行优化，对于计算第i个房屋时，有关的数据只有(i - 1) ~ i三组数据，可以每次只记录这两组数据，
# 并在转移后丢弃掉最早的，加上新的数据，进入下一个循环继续计算。空间复杂度为O(1) 。
class Solution:
    """
    @param A: An array of non-negative integers
    @return: The maximum amount of money you can rob tonight
    """
    # Method.1      二维DP
    def houseRobber(self, A):
        if A == []:
            return 0

        n = len(A)
        dp = [[0] * 2 for _ in range(n)]
        
        dp[0][0], dp[0][1] = 0, A[0]
        
        for i in range(1, n):
            # 如果不抢第 i 个，取前 i - 1 个位置 dp 较大值
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
            # 如果抢第 i 个，前一个不抢，考虑从前 i - 2 个位置的dp值转移
            dp[i][1] = A[i] + dp[i - 1][0]

        return max(dp[n - 1][0], dp[n - 1][1])


    
    # Method.2      一维DP
    def houseRobber(self, A):
        if not A:
            return 0
        if len(A) <= 2:
            return max(A)
            
        n = len(A)
        dp = [0] * (n + 1)
        dp[1] = A[0]
        
        for i in range(2, n+1):
            dp[i] = max(dp[i-2] + A[i-1], dp[i-1])
            
        return dp[-1]



    # Method.3      使用滚动数组的做法，将每个下标直接 % 3。时间复杂度 O(n)，空间复杂度 O(1)
    def houseRobber(self, A):
        if not A:
            return 0
        if len(A) <= 2:
            return max(A)
            
        f = [0] * 3
        f[0], f[1] = A[0], max(A[0], A[1])
        
        for i in range(2, len(A)):
            f[i % 3] = max(f[(i - 1) % 3], f[(i - 2) % 3] + A[i])
            
        return f[(len(A) - 1) % 3]



    # Method.4      DFS + Memorization, DP和记忆化搜索是一个东西, 虽然分别叫做Top-Down和Bottom-Up, 
    #               但本质都是现将问题化简到更小的子问题直至base case, 然后通过base case的值不断反推回去, 
    #               这个过程中cache子问题的最优解, 这样递归时可以直接返回
    def houseRobber(self, A):
        # write your code here
        self.steal_map = {}
        return self.dfs(A, 0)
    
    def dfs(self, A, index):
        if index in self.steal_map:
            return self.steal_map[index]
        if index >= len(A):
            return 0
            
        steal = A[index] + self.dfs(A, index + 2)
        not_steal = self.dfs(A, index + 1)
        
        self.steal_map[index] = max(steal, not_steal)
        return max(steal, not_steal)





# 534 · House Robber II
"""
After robbing those houses on that street, the thief has found himself a new place for his thievery so that 
he will not get too much attention. This time, all houses at this place are arranged in a circle. 
That means the first house is the neighbor of the last one. Each house holds a certain amount of money. 
The only constraint you face is that adjacent houses are equipped with interconnected anti-theft systems, 
which will automatically alarm when two adjacent houses are robbed on the same day.
Given a list of non-negative integers representing the amount of money of each house, determine the maximum 
amount of money you can rob tonight without alerting the police.

Input:  nums = [3,6,4]
Output: 6
"""
class Solution:
    """
    @param nums: An array of non-negative integers.
    @return: The maximum amount of money you can rob tonight
    """
    # Method.1      环形DP,这是392. 打劫房屋的拓展题，由原来的直线变成了圈，那么头尾两数变成了相邻的，不能取到，
    #               那么把这个圈分成[0,n-1]和[1,n]两部分分别计算当前能取到的最大值，最后将两部分的最大值比较，取较大值。
    #               用L[i]表示[0,n-1]部分在[0,i]内可以取到的最大值，用R[i]表示[1,n]部分在[1,i]内可以取到的最大值，
    #               从左往右遍历，最后两个快的最大值分别在L[size-1]和R[size-1]中, 最后的答案就是max(L[size-1],R[size-1])
    #               L[i] = max(L[i-2]+nums[i-1],L[i-1])
    #               R[i] = max(R[i-2]+nums[i],  R[i-1])
    #               时间复杂度O(n)  :       相当于遍历一遍nums
    #               空间复杂度O(n)  :       开两个数组记录当前最大值
    def house_robber2(self, nums) -> int:
        size = len(nums)
        if size == 0:
            return 0
        if size == 1:
            return nums[0]

        L = [0] * size
        R = [0] * size
        L[1] = nums[0]
        R[1] = nums[1]

        for i in range(2, size):
            #递推公式，注意nums中数的顺序先后
            L[i] = max(L[i - 2]+nums[i - 1] , L[i - 1])
            R[i] = max(R[i - 2]+nums[i] , R[i - 1])

        return max(L[size - 1], R[size - 1])


    # Method.2      考虑前若干个房子，记录抢最后一个房子或者不抢最后一个房子能抢到的最多的钱,然后交叉更新
    def houseRobber2(self, nums):
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]

        dp = [0] * n
        
        dp[0], dp[1] = 0, nums[1]
        for i in range(2, n):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        answer = dp[n - 1]

        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, n - 1):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return max(dp[n - 2], answer)





# 535 · House Robber III
"""
After robbing a street and a circle of houses last time, the burglar found a new place to rob. But this time, 
the area composed of all the houses is strange. After investigating the terrain, the clever burglar found that 
the terrain this time is a binary tree. Similar to the previous two thefts, each house had a certain amount of 
money in it. The only constraint you face is that adjacent houses are equipped with interconnected anti-theft 
systems, which will automatically alarm when two adjacent houses are robbed on the same day.
Calculate how much money you can get if you rob tonight, without touching the alarm.

Input:  {3,4,5,1,3,#,1}
Output: 9
Explanation:
Maximum amount of money the thief can rob = 4 + 5 = 9.
    3
   / \
  4   5
 / \   \ 
1   3   1
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

# 可以用树上的动态规划求解，对于每个节点，如果我们知道子节点中抢或不抢的子树中的最大值，就可以计算出当前节点抢或不抢的最大值。
# 且一个父节点的决策对于子节点抢或不抢的决策的最大值没有影响，这满足了动态规划的无后效性和最优子结构，所以可以用动态规划求解。
# 设 dp(node, 0)代表某个节点选择不抢时从它的子树中能抢的最大值。dp(node, 1)代表某个节点选择抢时他的子树中能抢的最大值。
# 可以得出状态转移方程：
# dp(node, 0) = max(dp(left, 0), dp(left, 1)) + max(dp(right, 0), dp(right, 1))
# dp(node, 1) = root.val + dp(left, 0) + dp(right, 0)
# 然后通过一次二叉树的后序遍历，先计算出子节点的值，就可以得到当前节点的值，自底向上更新。
# 复杂度分析    设树的节点数为N。
# 时间复杂度:   遍历一遍树，O(1)的时间状态转移，时间复杂度为O(N)。
# 空间复杂度:   遍历树的空间复杂度取决于树的最大深度，树可能的最大深度为N，空间复杂度为O(N)。
class Solution:
    """
    @param root: The root of binary tree.
    @return: The maximum amount of money you can rob tonight
    """
    def house_robber3(self, root: TreeNode) -> int:
        rob, not_rob = self.visit(root)
        return max(rob, not_rob)
        
    def visit(self, root):
        if root is None:
            return 0, 0
        
        left_rob, left_not_rob = self.visit(root.left)
        right_rob, right_not_rob = self.visit(root.right)
        
        # 根据左右子树的信息值，计算当前节点抢或不抢的最大值
        rob = root.val + left_not_rob + right_not_rob
        not_rob = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)

        return rob, not_rob