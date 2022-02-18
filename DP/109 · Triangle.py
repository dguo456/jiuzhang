# 109 · Triangle
"""
Given a triangle, find the minimum path sum from top to bottom. Each step you may move to 
adjacent numbers on the row below.
triangle = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
Output: 11
"""
import sys

class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    # Method.1  Traverse 无返回值  O(2^n) 会超时
    def minimumTotal(self, triangle):
        if not triangle or len(triangle) == 0 or len(triangle[0]) == 0:
            return 0

        self.result = sys.maxsize
        self.traverse(triangle, 0, 0, 0)
        return self.result

    def traverse(self, triangle, row, col, total):
        if row == len(triangle):
            self.result = min(total, self.result)
            return

        self.traverse(triangle, row+1, col, total+triangle[row][col])
        self.traverse(triangle, row+1, col+1, total+triangle[row][col])


    # Method.2      Divide&Conquer 带返回值   O(2^n) 会超时
    def minimumTotal(self, triangle):
        if not triangle or len(triangle) == 0 or len(triangle[0]) == 0:
            return 0

        return self.divdeConquer(triangle, 0, 0)

    def divdeConquer(self, triangle, row, col):
        if row == len(triangle):
            return 0

        left = self.divdeConquer(triangle, row+1, col)
        right = self.divdeConquer(triangle, row+1, col+1)

        return min(left, right) + triangle[row][col]


    # Method.3      dfs+memo记忆化剪枝  O(n^2)， 加cache统一套路
    def minimumTotal(self, triangle):
        if not triangle or len(triangle) == 0 or len(triangle[0]) == 0:
            return 0

        memo = {}
        return self.DFS_plus_memo(triangle, 0, 0, memo)

    def DFS_plus_memo(self, triangle, x, y, cache):
        if (x, y) in cache: return cache[(x,y)]
        if x == len(triangle): 
            return 0

        left = self.DFS_plus_memo(triangle, x+1, y, cache)
        right = self.DFS_plus_memo(triangle, x+1, y+1, cache)

        res =  min(left, right) + triangle[x][y]
        cache[(x,y)] = res
        
        return res


    # Method.4      DP  （填表格）    Time O(n^2), Space: O(n^2)
    def minimumTotal(self, triangle):
        if not triangle or len(triangle) == 0 or len(triangle[0]) == 0:
            return 0

        # dp[i][j]: minpathSum from top to triangle[i][j] 
        # dp[i][j] = min(dp[i-1][j-1], dp[i-1][j])+triangle[i][j]
        # initial dp[0][0] = triangle[0][0]，利用递推公式搞不出来就需要初始化
        # result: min(dp[n-1])
        n = len(triangle)
        dp = [[float('inf')] * (i+1) for i in range(n)]
        dp[0][0] = triangle[0][0]

        for r in range(1, n):
            for c in range(r+1):
                # most left edge, only one selection dp[r-1][c]
                if c == 0:
                    dp[r][c] = dp[r-1][c] + triangle[r][c]
                # most right edge, only one selection dp[r-1][c-1]
                elif c == r:
                    dp[r][c] = dp[r-1][c-1] + triangle[r][c]
                # in between
                else:
                    dp[r][c] = min(dp[r-1][c-1], dp[r-1][c]) + triangle[r][c]

        return min(dp[n-1])



    # Method.5      DP + rolling array 优化空间 （求余大法)  O(n^2) + O(n) space 优化空间, 求余大法好
    def minimumTotal(self, triangle):
        if not triangle or len(triangle) == 0 or len(triangle[0]) == 0:
            return 0

        n = len(triangle)
        dp = [[float('inf')] * n for _ in range(2)]
        dp[0][0] = triangle[0][0]

        r = 0
        for r in range(1, n):
            for c in range(r+1):
                if c == 0:
                    dp[r%2][c] = dp[(r-1)%2][c] + triangle[r][c]
                elif c == r:
                    dp[r%2][c] = dp[(r-1)%2][c-1] + triangle[r][c]
                else:
                    dp[r%2][c] = min(dp[(r-1)%2][c-1], dp[(r-1)%2][c])+triangle[r][c]
                    
        return min(dp[r%2])