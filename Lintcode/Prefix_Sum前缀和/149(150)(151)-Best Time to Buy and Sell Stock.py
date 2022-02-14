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




