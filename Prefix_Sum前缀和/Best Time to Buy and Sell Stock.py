# 149 · Best Time to Buy and Sell Stock
"""
Say you have an array for which the ith element is the price of a given stock on day i.
If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), 
design an algorithm to find the maximum profit.

Example 1
Input: [3, 2, 3, 1, 2]
Output: 1
Explanation: You can buy at the third day and then sell it at the 4th day. The profit is 2 - 1 = 1

Example 2
Input: [1, 2, 3, 4, 5]
Output: 4
Explanation: You can buy at the 0th day and then sell it at the 4th day. The profit is 5 - 1 = 4
"""
import sys

class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        total = 0
        lowest = sys.maxsize

        for price in prices:
            # if price - lowest > total:
            #     total = price - lowest
            # if price < lowest:
            #     lowest = price
            total = max(price - lowest, total)
            lowest = min(price, lowest)

        return total



# 150 · Best Time to Buy and Sell Stock II
"""
Given an array prices, which represents the price of a stock in each day.
You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). 
However, you may not engage in multiple transactions at the same time (ie, if you already have the stock, 
you must sell it before you buy again). Design an algorithm to find the maximum profit.

Input: [2, 1, 2, 0, 1]
Output: 2
Explanation: 
    1. Buy the stock on the second day at 1, and sell the stock on the third day at 2. Profit is 1.
    2. Buy the stock on the 4th day at 0, and sell the stock on the 5th day at 1. Profit is 1.
    Total profit is 2.
"""

# 贪心法: 只要相邻的两天股票的价格是上升的, 我们就进行一次交易, 获得一定利润.
# 这样的策略不一定是最小的交易次数, 但是一定会得到最大的利润.
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        profit = 0

        for i in range(1, len(prices)):
            if prices[i] - prices[i - 1] > 0:
                profit += prices[i] - prices[i - 1]

        return profit


    # Method.2      DP
    def maxProfit_DP(self, prices):
        # 允许买卖无数次，但是手中只能持有单只股票。
        if bool(prices) == False or len(prices) < 2:
            return 0
        maxProfit, length = 0, len(prices)
        profit = [[0 for _ in range(2)] for _ in range(length)]          
        profit[0][0], profit[0][1] = 0, -prices[0]               #记录第一天是否购买了股票
        
        """
        dp方程:     dp[i][j] i维信息定义记录到第i天的利润；j维信息为0或1，用来标记是否持有股票。
                    dp[i][0] = max{ dp[i - 1][0]                  前天没有股票，维持不动
                                    dp[i - 1][1] + prices[i]     前天有股票，今天卖出 }
                    dp[i][1] = max{ dp[i - 1][1],                前天有股票，维持不动
                                    dp[i - 1][0] - prices[i]}    前天没有股票，买入股票
        """
        for day in range(1, length):
            profit[day][0] = max(profit[day - 1][0], profit[day - 1][1] + prices[day])
            profit[day][1] = max(profit[day - 1][1], profit[day - 1][0] - prices[day])
            maxProfit = max(maxProfit, profit[day][0], profit[day][1])

        return maxProfit




# 151 · Best Time to Buy and Sell Stock III
"""
Say you have an array for which the ith element is the price of a given stock on day i.
Design an algorithm to find the maximum profit. You may complete at most two transactions.

Input : [4,4,6,1,1,4,2,5]
Output : 6
"""

# Method.1  隔板法，同 Subarray系列 第42题的解法
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        n = len(prices)
        if n <= 1:
            return 0
            
        p1 = [0] * n
        p2 = [0] * n
        
        minV = prices[0]
        for i in range(1, n):
            minV = min(minV, prices[i]) 
            p1[i] = max(p1[i - 1], prices[i] - minV)
        
        maxV = prices[-1]
        for i in range(n-2, -1, -1):
            maxV = max(maxV, prices[i])
            p2[i] = max(p2[i + 1], maxV - prices[i])
        
        res = 0
        for i in range(n):
            res = max(res, p1[i] + p2[i])
        return res




# 393 · Best Time to Buy and Sell Stock IV
"""
Given an array prices and the i-th element of it represents the price of a stock on the i-th day.
You may complete at most k transactions. What's the maximum profit?

Input: k = 2, prices = [4, 4, 6, 1, 1, 4, 2 ,5]
Output: 6
Explanation: Buy at 4 and sell at 6. Then buy at 1 and sell at 5. Your profit is 2 + 4 = 6.
"""

# 对于动态规划问题，首先要明确两点：状态和选择。然后就可以按照这个框架进行套用
# 先举个经典的例子，背包问题。我们的状态有「背包的容量」和「可选择的物品」。对于每件物品，选择就是「装进背包」或者「不装进背包」。
# 所以可以定义二维数组dp[N][W]。任意元素dp[i][w]的含义是：对于前i个物品，当前背包的容量为w，这种情况下可以装的最大价值。
# dp[i][w] = max(把物品 i 装进背包, 不把物品 i 装进背包)

# 本题我们的目标是求解最大利润，其中有两个状态属性：「交易最大次数k」和「天数n」。
# 当天面临的选择也很简单：「不做处理」和「卖出股票」，暂时先不考虑何时买入的问题。
# 我们定义dp[k + 1][n]，元素dp[i][j]表示最多交易i次时在第j天能获得的最大利润。dp[i][j] = max(不做处理， 卖出股票)
# 如果不做处理，第j天的最大利润就和第j-1天相等。
# 如果卖出股票，设是在第m天买入，那么第j天的最大利润就是两天的价格差+最多交易i-1次时第m天的获利。当然，这里的m需要从0遍历到j-1。
# 观察上述公式，会发现在计算dp[i][j]和dp[i][j+1]时，向前遍历寻找最合适的买入天（即m）的工作是大部分重合的。
# 如果我们在遍历dp数组中实时更新一个变量maxDiff，就能节省这部分的时间开销。maxDiff表示：最多交易i-1次时，
# 从第0天到第j-1天最大利润-当天价格的最大值。

# 时间复杂度: O(n * k)。遍历dp的每个元素。
# 空间复杂度: O(n * k)。考虑dp数组占用的空间。
# 当k >= n / 2时，可退化为无限次交易的问题，我们单独处理这种corner case。
# 空间复杂度也可以继续优化，因为我们在考虑最多进行i次交易时，只用到了dp[i-1]的数据，所以可以只需要保存两行。空间复杂度可以优化成O(n)。

class Solution:
    """
    @param K: An integer
    @param prices: An integer array
    @return: Maximum profit
    """
    def maxProfit(self, K, prices):
        n = len(prices)
        # corner case
        if n == 0 or K == 0:
            return 0
        # corner case: equal to infinity times of transaction
        if K >= n // 2:
            res = 0
            for i in range(1, n):
                res += max(0, prices[i] - prices[i - 1])
            return res

        # main part
        dp = [[0] * n for _ in range(K + 1)]
        for i in range(1, K + 1):
            max_diff = float('-inf')
            for j in range(1, n):
                max_diff = max(max_diff, dp[i - 1][j - 1] - prices[j - 1])
                dp[i][j] = max(dp[i][j - 1], prices[j] + max_diff)
                
        return dp[K][n - 1]




# 1691 · Best Time to Buy and Sell Stock V
"""
Given a stock n-day price, you can only trade at most once a day, you can choose to buy a stock or 
sell a stock or give up the trade today, output the maximum profit can be achieved.

Given `a = [1,2,10,9]`, return `16`
Input:  [1,2,10,9]
Output: 16
Explanation:
you can buy in day 1,2 and sell in 3,4.     profit:-1-2+10+9 = 16 
"""

# 算法： 优先队列
# 从前往后遍历每天的股票价格，我们考虑每天能够获取的最大收益，考虑贪心获得最大收益，那么我们可以从左往右遍历，
# 若当前价格大于之前遇到的最低价，则做交易。同时把在heap里用卖出价代替买入价，即将当前价格压入队列（假设当前价格为b,
# 要被弹出的元素是a,后面有一个c元素，如果那时a还在，作为最低价，差值为c-a，而这里已经被b拿去做了差值，所以b得压入队列，
# 因为c-b+b-a = c-a），弹出之前的最低价,可以利用优先队列来使之前的价格有序.
# 用优先队列存储当前遇到过的价格, 每日的新价格 与历史最低价比较, 若比最低价高，则弹出最低价，同时更新答案，即加上差值,压入当前价格
# 复杂度分析
# 时间复杂度O(nlogn)    n为天数,优先队列的复杂度
# 空间复杂度O(n)        n为天数
import heapq

class Solution:
    """
    @param a: the array a
    @return: return the maximum profit
    """
    def getAns(self, a):
        minheap = []
        result = 0

        for k in a:
            # 如果k比之前遇到过的最低价高
            if minheap and k > minheap[0]:
                # 收益就是当前k - 遇到过的最低价
                result += k - heapq.heappop(minheap)
                heapq.heappush(minheap, k)

            # 同时将当前值压入队列
            heapq.heappush(minheap,k)

        return result