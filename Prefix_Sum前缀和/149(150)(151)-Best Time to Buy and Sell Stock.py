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