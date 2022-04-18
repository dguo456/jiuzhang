# 392 · House Robber
# 534 · House Robber II
# 535 · House Robber III
# 514 · Paint Fence
# 515 · Paint House
# 516 · Paint House II
# 866 · Coin Path

################################################################################################################

import sys

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
# 这满足了动态规划的 "无后效性" 和 "最优子结构"。同时，由于题目不能抢相邻房屋，那么如果抢了第i个房屋，
# 就不能抢第i - 1个房屋，可以得出前i个的最优方案也与前i - 2个的最优方案有关。
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
    # Method.1      构建一个一维DP数组，需要dp[0]来存储初始状态。时间复杂度 O(n)，空间复杂度 O(n)
    def houseRobber(self, A):
        if not A:
            return 0
        if len(A) <= 2:
            return max(A)

        # 初始化成 len(A)的长度
        dp = [0] * len(A)
        dp[0], dp[1] = A[0], max(A[0], A[1])
        
        for i in range(2, len(A)):
            dp[i] = max(dp[i - 2] + A[i], dp[i - 1])
            
        return dp[-1]



    # Method.2      二维DP (实际也是一维数组，只不过用两个值分别代表抢与不抢)
    def houseRobber(self, A):
        if A == []:
            return 0

        n = len(A)
        dp = [[0] * 2 for _ in range(n)]
        
        dp[0][0], dp[0][1] = 0, A[0]
        
        for i in range(1, n):
            # 如果不抢第 i 个，取 dp[i-1] 两个值当中的较大值
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
            # 如果抢第 i 个，前一个必须不抢，考虑从前 i - 2 个位置的dp值转移
            dp[i][1] = A[i] + dp[i - 1][0]

        return max(dp[n - 1][0], dp[n - 1][1])



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
    #               这个过程中cache子问题的最优解, 这样递归时可以直接返回 (此解法会超时)
    def houseRobber(self, A):
        if not A:
            return 0

        memo = {}
        return self.dfs(A, 0, memo)

    def dfs(self, A, index, memo):
        if index >= len(A):
            return 0

        if index in memo:
            return memo[index]

        rob = self.dfs(A, index + 2, memo) + A[index]
        not_rob = self.dfs(A, index + 1, memo)

        memo[index] = max(rob, not_rob)
        
        return max(rob, not_rob)





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
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)

        n = len(nums)
        dp = [0] * n

        # Rob the last house
        dp[0], dp[1] = 0, nums[1]
        for i in range(2, n):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        rob = dp[-1]

        # not rob the last house
        dp[0], dp[1] = nums[0], max(nums[0], nums[1])
        for i in range(2, n-1):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        not_rob = dp[-2]

        return max(rob, not_rob)





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
# dp(node, 0) = max(dp(left, 0), dp(left, 1))  +  max(dp(right, 0), dp(right, 1))
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





# 514 · Paint Fence
"""
There is a fence with n posts, each post can be painted with one of the k colors.
You have to paint all the posts such that no more than two adjacent fence posts have the same color.
Return the total number of ways you can paint the fence.

Input: n=2, k=2  
Output: 4
Explanation:
          post 1,   post 2
    way1    0         0       
    way2    0         1            
    way3    1         0          
    way4    1         1       
"""

class Solution:
    """
    @param n: non-negative integer, n posts
    @param k: non-negative integer, k colors
    @return: an integer, the total number of ways
    """
    # Method 1: DP without Rolling Array
    # dp[i][j] denotes total number of ways to paint i posts by using in total j colors
    # dp[i][j] = case_1 + case_2
    # case_1: if painting the ith post is using different color of (i - 1)th, case_1 = dp[i - 1] * (k - 1) 
    # case_2: if painting the ith post is using same color of (i - 1)th post, case_2 = dp[i - 2] * (k - 1)
    def numWays(self, n, k):
        if not n or not k or (k == 1 and n > 2):
            return 0
        if n == 1:
            return k
        if n == 2:
            return k * k

        dp = [0] * n
        dp[0], dp[1] = k, k * k

        for i in range(2, n):
            dp[i] = dp[i-1] * (k-1) + dp[i-2] * (k-1)

        return dp[-1]


    
    # Method 2: DP with Rolling Array
    def numWays(self, n, k):
        if not n or not k or (k == 1 and n > 2):
            return 0
        if n == 1:
            return k
        if n == 2:
            return k * k
        
        dp = [0] * 2 
        dp[0], dp[1] = k, k * k

        for _ in range(n - 2):
            dp[0], dp[1] = dp[1], dp[0] * (k - 1) + dp[1] * (k - 1)

        return dp[1]





# 515 · Paint House
"""
There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. 
The cost of painting each house with a certain color is different. You have to paint all the houses 
such that no two adjacent houses have the same color, and you need to cost the least. Return the minimum cost.
The cost of painting each house with a certain color is represented by a n x 3 cost matrix. 
For example, costs[0][0] is the cost of painting house 0 with color red; costs[1][2] is the cost of 
painting house 1 with color green, and so on... Find the minimum cost to paint all houses.

Input: [[14,2,11],[11,14,5],[14,3,10]]
Output: 10
Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue. 
Minimum cost: 2 + 5 + 3 = 10.
"""

# dp[i][j]意思是第i个房子涂成j颜色的情况下从第一间房到i间房一共需要花钱的最小值，其中 j取值范围是(0, 1, 2)对应(r, g, b)
# 状态转移方程：
# dp[i][0] = min(dp[i - 1][1], dp[i - 1][2]) + cost[0]
# dp[i][1] = min(dp[i - 1][0], dp[i - 1][2]) + cost[1]
# dp[i][2] = min(dp[i - 1][0], dp[i - 1][1]) + cost[2]
class Solution:
    """
    @param costs: n x 3 cost matrix
    @return: An integer, the minimum cost to paint all houses
    """
    def min_cost(self, costs) -> int:
        if not costs:
            return 0

        dp = [[float('inf')] * 3 for _ in costs]

        for i, cost in enumerate(costs):
            if i == 0:
                dp[i] = cost[:]
                continue

            # red, blue, green
            dp[i][0] = min(dp[i-1][1], dp[i-1][2]) + cost[0]
            dp[i][1] = min(dp[i-1][0], dp[i-1][2]) + cost[1]
            dp[i][2] = min(dp[i-1][0], dp[i-1][1]) + cost[2]

        return min(dp[-1])


    # Method.2      用到了滚动数组和python里的generator，并且延伸到了多种颜色的情况。
    #               空间复杂度：O(1) / 时间复杂度：O(N)
    def minCost(self, costs):
        if not costs or not costs[0]:
            return 0

        INF = 0x7fffffff
        n, m = len(costs), len(costs[0])
        dp = [costs[0], [None for _ in range(m)]]

        for i in range(1, n):
            for j in range(m):
                dp[i%2][j] = INF
                for k in range(m):
                    if k != j:
                        dp[i%2][j] = min(dp[i%2][j], dp[(i-1)%2][k] + costs[i][j])

        return min(dp[(n-1)%2])


    # Method.3
    def minCost(self, costs) -> int:
        r, b, g = 0, 0, 0

        for r_c, b_c, g_c in costs:
            r, b, g = r_c + min(b, g), b_c + min(r, g), g_c + min(r, b)

        return min(r, b, g)


    # Method.4      DFS + memo  思路有点类似triangle的解法，下一层只能左右。
    def minCost(self, costs):
        if not costs:
            return 0

        result = self.helper(costs, -1, -1, {})
        return result
        
    def helper(self, costs, pos, color, memo):
        if (pos, color) in memo:
            return memo[(pos, color)]
            
        if pos == len(costs):
            return 0 
        
        min_cost = sys.maxsize
        curr_cost = costs[pos][color] if pos >= 0 else 0
        
        for next_color in range(3):
            if next_color == color:
                continue 
            
            next_cost = self.helper(costs, pos + 1, next_color, memo) + curr_cost
            min_cost = min(min_cost, next_cost)
        
        memo[(pos, color)] = min_cost
        return min_cost






# 516 · Paint House II
"""
There are a row of n houses, each house can be painted with one of the k colors. The cost of painting 
each house with a certain color is different. You have to paint all the houses such that no two 
adjacent houses have the same color.
The cost of painting each house with a certain color is represented by a n x k cost matrix. 
For example, costs[0][0] is the cost of painting house 0 with color 0; 
costs[1][2] is the cost of painting house 1 with color 2, and so on... 
Find the minimum cost to paint all houses.

Input:
costs = [[14,2,11],[11,14,5],[14,3,10]]
Output: 10
Explanation:
The three house use color [1,2,1] for each house. The total cost is 10.
"""
# dp[i][j]表示第i幢房子涂j的颜色最小的总和, 即从前一幢房子的状态dp[i-1][k] (k != j)中选一个不同颜色且最小的
# 再加上给第i幢房子涂j颜色的costs。状态转移方程是dp[i][j] = min{dp[i-1][h] +costs[i][j]} (h != j)
# 这里的颜色数为k，远大于515的3种颜色，因此如果我们对于每一个dp[i][j]枚举了每一个dp[i-1][h]，那么总的时间复杂度高达O(N*K^2)
# 我们计算每一层dp[i]时，可以先计算好dp[i-1]中的最小值，这样就可以在O(1)的时间转移每个状态；
# 但是如果最小值和现在要涂的颜色相同怎么办，因此我们还要计算好dp[i-1]中的次小值，如果最小值和现在要涂的颜色相同，
# 那么就用次小值来转移。这样总的时间复杂度就变成了O(NK). 滚动存储状态，可以将空间复杂度从O(NK)优化到O(K)。
# 从左往右遍历每一幢房子，计算到当前幢房子时，先计算到前一幢房子的最小花费和次小花费，转移时若当前涂的颜色与前一幢房子
# 的最小值涂的颜色不同，那么到当前房子涂此种颜色的最小花费就是前一幢房子的最小花费加上涂此种颜色的花费，
# 否则就是前一幢房子的次小花费加上涂此种颜色的花费. 空间复杂度：O(K)/时间复杂度：O(NK)

# 九章给的解法很巧妙和精简，巧妙利用了“The cost of painting each house with a certain color is different”这个限制，
# 只需保存一个最小cost min1和一个次小cost min2. 所以在第二个内循环里才可以比较f[old][j] == min1。
class Solution:
    """
    @param costs: n x k cost matrix
    @return: an integer, the minimum cost to paint all houses
    """
    def minCostII(self, costs):
        if not costs or not costs[0]:
            return 0
        
        n, k = len(costs), len(costs[0])

        # DP_1: state + DP_2: init
        dp = [[0] * k for _ in range(n + 1)]

        for i in range(1, n + 1):
            # find the index of min and the second min among dp[i - 1][0], ..., dp[i - 1][k - 1]
            min1, min2 = -1 , -1
            for j in range(k):
                if min1 == -1 or dp[i - 1][j] < dp[i - 1][min1]:
                    min2 = min1
                    min1 = j
                else:
                    if min2 == -1 or dp[i - 1][j] < dp[i - 1][min2]:
                        min2 = j

            # DP_3: shift function
            for j in range(k):
                if j != min1:
                    dp[i][j] = costs[i - 1][j] +  dp[i - 1][min1]
                else:
                    dp[i][j] = costs[i - 1][j] +  dp[i - 1][min2]
        
        # DP_4: answer
        return min(dp[-1][:])



# Method.2
# 分析 dfs + memoization 的三种解法，逐步优化
# dp如果想不出来，dfs + memoization 是比较保险的。
import heapq

class Solution:
    """
    @param costs: n x k cost matrix
    @return: an integer, the minimum cost to paint all houses
    """
    # 第一种解法，暴力dfs + memoization
    # 每次递推一个house index，记录上一个的color。每次遍历所有的color，除了上一个color，都继续向下递归搜索。
    # 最后时间: 递归次数 * 单次递归计算时间 = O(nk^2) ，在比较大的测试case中达到 O(600 * 400^2) == 10^9 级别, TLE
    def minCostII(self, costs):
        if not costs or not costs[0]:
            return 0

        return self.dfs(costs, 0, -1, {})

    def dfs(self, costs, house_index, prev_color, memo):
        if house_index == len(costs):
            return 0

        if (house_index, prev_color) in memo:
            return memo[(house_index, prev_color)]

        result = float('inf')
        for color in range(len(costs[0])):
            if color == prev_color:
                continue
            result = min(
                result,
                self.dfs(costs, house_index+1, color, memo) + costs[house_index][color]
            )

        memo[(house_index, prev_color)] = result

        return result


    # 第二种解法，剪枝
    # 分析之后发现重要性质，每次是否需要遍历所有的color都向下递归？其实不用，只需要递归2次。cost最小的color和cost次小的color。
    # 因为：只需要保证如果下一层递归color和2种之一如果一样，有另一种可用即可。（这里想明白之后就基本可以AC了，不会爆时间。）
    # 最后时间：O(nk^2)，但是会剪枝，能AC，beats 10% 左右
    def minCostII(self, costs):
        if not costs or not costs[0]:
            return 0

        return self.dfs(costs, 0, -1, {})

    def dfs(self, costs, house_index, prev_color, memo):
        if house_index == len(costs):
            return 0

        if (house_index, prev_color) in memo:
            return memo[(house_index, prev_color)]

        result = float('inf')
        heap = []
        for color in range(len(costs[0])):
            if color == prev_color:
                continue
            heapq.heappush(heap, (-costs[house_index][color], color))
            if len(heap) > 2:
                heapq.heappop(heap)

        for _, color in heap:
            result = min(
                result,
                self.dfs(costs, house_index+1, color, memo) + costs[house_index][color]
            )

        memo[(house_index, prev_color)] = result

        return result


    # 第三种解法，预处理
    # 由第二种方法发现，每次递归都需要遍历 k 种color去找到最小和第二小，此处计算重复。
    # 程序开始的时候对每个房子都预处理，提前拿到最小和第二小。递归的时候可以通过hash，用 O(1) 的时间，拿到要向下递归的color。
    # 最后时间：O(2nk)，beats 30~40%
    def minCostII(self, costs):
        if not costs:
            return 0

        self.house_min_cost = [None] * len(costs)
        for i in range(len(costs)):
            heap = []
            house = costs[i]
            for color in range(len(house)):
                heapq.heappush(heap, (-house[color], color))
                if len(heap) > 3:
                    heapq.heappop(heap)
            self.house_min_cost[i] = heap

        return self.dfs(costs, 0, -1, {})

    def dfs(self, costs, house_index, prev_color, memo):
            if house_index == len(costs):
                return 0

            if (house_index, prev_color) in memo:
                return memo[(house_index, prev_color)]

            result = float('inf')

            for _, color in self.house_min_cost[house_index]:
                if color == prev_color:
                    continue
                result = min(
                    result,
                    self.dfs(costs, house_index + 1, color, memo) + costs[house_index][color]
                )

            memo[(house_index, prev_color)] = result
            return result






# 866 · Coin Path
"""

"""
class Solution:
    """
    @param A: a list of integer
    @param B: an integer
    @return: return a list of integer
    """
    # 由于路径上正向和反向考虑均可，此处采用反向，dp[i]表示从i位置后到达i位置的最小代价。
    # i采用逆序，枚举i位置后B范围内的位置j. 由于从靠近i的位置开始转移，所以字典序已经满足最小。
    def cheapestJump(self, A, B):
        n = len(A)
        # a = b = c，只适用于a，b，c均为常数Integer
        dp , nxt = [-1 for _ in range(0 , n)] , [-1 for _ in range(0 , n)]
        dp[n - 1] = A[n - 1]

        for i in range (n - 2, -1 ,-1):       #i逆序枚举
            if A[i] != -1 :
                for j in range(i + 1, min(i+B+1, n)):   #枚举i位置后B范围内的j
                    if dp[j] != -1 :
                        if dp[i] == -1 or dp[j] + A[i] < dp[i]:
                            dp[i] = dp[j] + A[i]
                            nxt[i] = j					#将j作为i的下一个节点
        if dp[0] == -1 :
            return []

        now = 0
        path = []
        while now != -1 :						#输出路径
            path.append(now + 1)
            now = nxt[now]

        return path