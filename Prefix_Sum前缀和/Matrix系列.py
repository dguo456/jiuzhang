# 110 · Minimum Path Sum        (对比1840)
"""
Given a m * n grid filled with non-negative numbers, find a path from top left to bottom right which 
minimizes the sum of all numbers along its path.
Input:  grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation:    Path is: 1 -> 3 -> 1 -> 1 -> 1
"""
from typing import (
    List,
)

class Solution:
    """
    @param grid: a list of lists of integers
    @return: An integer, minimizes the sum of all numbers along its path
    """
    # Method.1      动态规划记录从上到下的路径最小和，一个点只能从左边或者上边过来
    def min_path_sum(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0

        m, n = len(grid), len(grid[0])
        dp = [[float('inf') for _ in range(n)] for _ in range(m)]
        dp[0][0] = grid[0][0]

        for r in range(m):
            for c in range(n):
                if r == 0 and c > 0:
                    dp[r][c] = dp[r][c-1] + grid[r][c]
                elif c == 0 and r > 0:
                    dp[r][c] = dp[r-1][c] + grid[r][c]
                elif r > 0 and c > 0:
                    dp[r][c] = min(dp[r-1][c], dp[r][c-1]) + grid[r][c]

        return dp[m-1][n-1]


    # Method.2      时间复杂度：O(n*m)  二维dp遍历网络，需要两重循环，因此为O(n*m)；空间复杂度：O(n*m)
    def minPathSum(self, grid):
        if not grid:
            return 0

        m, n = len(grid), len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]

        # 初始化
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]

        # 状态转移
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1])+grid[i][j]

        return dp[m - 1][n - 1]





# 1582 · Minimum Path Sum II
"""
Given an n * m matrix, each point has a weight, from the $ bottom left $ of the matrix to the $ top right $
(you can go in four directions), let you find a path so that the path through the minimum sum of weights 
and return the minimum sum.
Input:  [[2,3][3,2]]
Output: 8
Explanation:    (1,0)->(1,1)->(0,1),the minimum sum is 8.
"""
from typing import (
    List,
)
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    """
    @param matrix: a matrix
    @return: the minimum height
    """
    def min_path_sum_i_i(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
            
        self.n, self.m = len(matrix), len(matrix[0])
        distance = [[float('inf') for _ in range(self.m)] for _ in range(self.n)]
        distance[self.n - 1][0] = matrix[self.n - 1][0]

        self.dfs(distance, matrix, matrix[self.n - 1][0], self.n-1, 0)
        return distance[0][self.m - 1]

    def dfs(self, distance, matrix, curr_sum, x, y):
        distance[x][y] = curr_sum
        for dx, dy in DIRECTIONS:
            next_x = x + dx
            next_y = y + dy

            if not self.is_valid(distance, matrix, curr_sum, next_x, next_y):
                continue
            
            self.dfs(distance, matrix, curr_sum + matrix[next_x][next_y], next_x, next_y)

    def is_valid(self, distance, matrix, curr_sum, x, y):
        if x < 0 or x >= self.n or y < 0 or y >= self.m:
            return False
        if distance[x][y] < curr_sum + matrix[x][y]:
            return False

        return True






# 1840 · Matrix restoration
"""
There is a matrix beforebefore with nn rows and mm columns. For each element in before[i][j], 
we will use the following algorithm to convert it to after[i][j]. Given the after matrix, 
please restore the original matrix before.
s = 0
for i1: 0 -> i
    for j1: 0 -> j
        s = s + before[i1][j1]
after[i][j] = s

Example:
Input:  2   2   [[1,3],[4,10]]
Output:  [[1,2],[3,4]]
Explanation:
before:
1 2
3 4

after:
1 3
4 10
"""

# Method.1
# 由题意可知，after[i][j]代表矩阵从(0, 0)到(i, j)的所有数的和。
# 如果我们想知道before[i][j]的值，就要用after[i][j]减去(0, 0)到(i - 1, j)的值，也要减去(0, 0)到(i, j - 1)的值。
# 但是其实(0, 0)到(i - 1, j - 1)被重复减了两次，需要再加回来一次。
# 所以可以得到递推公式  before[i][j] = after[i][j] - after[i-1][j] - after[i][j-1] + after[i-1][j-1]
# 可以直接在原矩阵上进行计算，从after矩阵的(n - 1, m - 1)从下往上，从右往左进行计算。
class Solution:
    """
    @param n: the row of the matrix
    @param m: the column of the matrix
    @param after: the matrix
    @return: restore the matrix
    """
    def matrixRestoration(self, n, m, after):
        # 倒序遍历矩阵
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                # 减去上面的部分
                if i > 0:
                    after[i][j] -= after[i - 1][j]
                # 减去左面的部分
                if j > 0:
                    after[i][j] -= after[i][j - 1]
                # 加上重复减去的部分
                if i > 0 and j > 0:
                    after[i][j] += after[i - 1][j - 1]
        
        return after



# Method.2      在after矩阵里的上方,左方填入0,来避免判断边界
class Solution:
    """
    @param n: the row of the matrix
    @param m: the column of the matrix
    @param after: the matrix
    @return: restore the matrix
    """
    def matrixRestoration(self, n, m, after):
        before = [[0 for _ in range(m)] for _ in range(n)] 
       
        after.insert(0, [0] * m)
        for row in after:
            row.insert(0, 0)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                before[i-1][j-1] = after[i][j] - after[i][j-1] - after[i-1][j] + after[i-1][j-1]

        return before