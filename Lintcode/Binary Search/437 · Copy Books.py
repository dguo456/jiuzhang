# 437 · Copy Books
"""
Given n books and the i-th book has pages[i] pages. There are k persons to copy these books.

These books list in a row and each person can claim a continous range of books. For example, 
one copier can copy the books from i-th to j-th continously, but he can not copy the 1st book, 
2nd book and 4th book (without 3rd book).

They start copying books at the same time and they all cost 1 minute to copy 1 page of a book. 
What's the best strategy to assign books so that the slowest copier can finish at earliest time?
Return the shortest time that the slowest copier spends.

Input: pages = [3, 2, 4], k = 2
Output: 5
Explanation: 
    First person spends 5 minutes to copy book 1 and book 2.
    Second person spends 4 minutes to copy book 3.
"""

# 可以使用二分或者动态规划解决这道题目. 不过更推荐二分答案的写法, 它更节省空间, 思路简洁, 容易编码.
# 对于假定的时间上限 tm 我们可以使用贪心的思想判断这 k 个人能否完成复印 n 本书的任务: 
# 将尽可能多的书分给同一个人, 判断复印完这 n 本书需要的人数是否不大于 k 即可.
# 而时间上限 tm 与可否完成任务(0或1)这两个量之间具有单调性关系, 所以可以对 tm 进行二分查找, 查找最小的 tm, 使得任务可以完成.
# 使用九章算法强化班中讲过的基于答案值域的二分法。
# 答案的范围在 max(pages)~sum(pages) 之间，每次二分到一个时间 time_limit 的时候，用贪心法从左到右
# 扫描一下 pages，看看需要多少个人来完成抄袭。如果这个值 <= k，那么意味着大家花的时间可能可以再少一些，
# 如果 > k 则意味着人数不够，需要降低工作量。

# 时间复杂度 O(nlog(sum)) 是该问题时间复杂度上的最优解法
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        if not pages:
            return 0

        start, end = max(pages), sum(pages)
        while start + 1 < end:
            mid = (start + end) // 2
            if self.get_least_people(pages, mid) <= k:
                end = mid
            else:
                start = mid

        if self.get_least_people(pages, start) <= k:
            return start
        return end

    def get_least_people(self, pages, time_limit):
        count = 0
        time_cost = 0

        for page in pages:
            # 这里注意，因为题目要求必须是连续copy
            if time_cost + page > time_limit:
                count += 1
                time_cost = 0
            time_cost += page

        return count + 1



# 采用暴力的动态规划算法，时间复杂度 O(n^2k)
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """

    def copyBooks(self, pages, k):
        if not pages or not k:
            return 0
        
        n = len(pages)
        # get prefix sum
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + pages[i - 1]
        
        # state: dp[i][j] 表示前 i 本书，划分给 j 个人抄写，最少需要耗费多少时间
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        
        # initialization
        for i in range(k + 1):
            dp[0][i] = 0

        # function
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for prev in range(i):
                    cost = prefix_sum[i] - prefix_sum[prev]
                    dp[i][j] = min(dp[i][j], max(dp[prev][j - 1], cost))

        return dp[n][k]