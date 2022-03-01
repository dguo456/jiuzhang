# 114 · Unique Paths
"""
A robot is located at the top-left corner of a m x nmxn grid.
The robot can only move either down or right at any point in time. The robot is trying to reach 
the bottom-right corner of the grid. How many possible unique paths are there?

Input:  n = 3   m = 3
Output: 6
"""
class Solution:
    """
    @param m: positive integer (1 <= m <= 100)
    @param n: positive integer (1 <= n <= 100)
    @return: An integer
    """
    def uniquePaths(self, m, n):
        
        dp = [[0] * n for i in range(m)]
        dp[0][0] = 1
        
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    dp[i][j] = 1
                    continue
                
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
                
        return dp[m-1][n-1]


    # Method.2      DFS + memo 记忆化搜索
    DIRECTIONS = [(1, 0), (0, 1)]
    def uniquePaths(self, m, n):
        memo = {}
        result = self.TotalPaths(0, 0, m, n, memo)
        return result
    
    def TotalPaths(self, row, column, m, n, memo):
        if (row, column) in memo:
            return memo[(row, column)]

        if row == m - 1 or column == n - 1:
            memo[(row, column)] = 1
            return memo[(row, column)]

        memo[(row, column)] = 0

        for direction in self.DIRECTIONS:
            memo[(row, column)] += self.TotalPaths(row + direction[0], column + direction[1], m, n, memo)
        
        return memo[(row, column)]





# 115 · Unique Paths II
"""
Follow up for "Unique Paths":
Now consider if some obstacles are added to the grids. How many unique paths would there be?
An obstacle and empty space is marked as 1 and 0 respectively in the grid.
"""

# 对于障碍物我们需要分两种情况考虑，分别是处于网格边界和网格中央时的情况，根据题意很容易发现处于边界的障碍物，
# - 左边界的第一个障碍物下面的所有边界点无法到达，
# - 上边界的第一障碍物右边的所有边界点无法到达。
# 对于网格中的障碍物，到达此点的路径条数默认为0。用数组dp表示可到达每个点的路径数
# 通过所有到达(i-1,j)这个点的路径往下走一步可到达(i,j), 这路径数总共有dp[i-1][j]条
# 通过所有到达(i,j-1)这个点的路径往右走一步可到达(i,j), 这路径数总共有dp[i][-1j]条
# 由此可以推出递推式dp[i][j] = dp[i-1][j]+dp[i][j-1], 如果数组长宽都为0的话返回0
# 设dp数组的大小与obstacleGrid数组大小一致, 对于左边界，在第一个障碍物前面（或者到达边界）的所有点皆可到达
# 对于上边界，在第一个障碍物左边（或者到达边界）的所有点皆可到达
# 从dp[1][1]开始遍历网格，根据递推式dp[i][j] = dp[i-1][j]+dp[i][j-1]更新当前点可到达路径数

# 时间复杂度O(n*m)      遍历一遍网格，复杂度即网格大小
# 空间复杂度O(n*m)      需要开一个数组记录当前路径数量
class Solution:
    """
    @param obstacleGrid: A list of lists of integers
    @return: An integer
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        n,m = len(obstacleGrid),len(obstacleGrid[0])
        if n == 0 and m == 0:
            return 0

        dp=[[0] * m for _ in range(n)]
        if obstacleGrid[0][0] == 0:
            dp[0][0] = 1

        for i in range(0,n):
            for j in range(0,m):
                if i == 0 and j == 0:
                    continue
                #若遇到障碍物，则跳过
                if obstacleGrid[i][j] == 1:
                    continue
                #对于上边界，第一个障碍物或边界左边的所有边界点皆可到达
                if i == 0:
                    dp[i][j] = dp[i][j-1]
                    continue
                #对于左边界，第一个障碍物或边界前的所有边界点皆可到达
                if j == 0:
                    dp[i][j] = dp[i-1][j]
                    continue
                #到达当前点的路径数等于能到达此点上面的点和左边点的路径数之和
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[n-1][m-1]





# 679 · Unique Paths III
"""
Follow up for "Unique Paths II", Now each grid contains a value, so each path also has a value. 
Find the sum of all the unique values paths.
Input:
[
  [1,1,2],
  [1,2,3],
  [3,2,4]
]
Output:
21
Explanation:
There are 2 unique value path:
[1,1,2,3,4] = 11
[1,1,2,2,4] = 10
"""
class Solution:
    """
    @param: : an array of arrays
    @return: the sum of all unique weighted paths
    """
    # Method.1      记忆化搜索
    def uniqueWeightedPaths(self, grid):
        if not grid or not grid[0]:
            return 0

        return sum(self.helper(0, 0, grid, {}))
    
    def helper(self, row, col, grid, memo):
        if row >= len(grid) or col >= len(grid[0]):
            return set()
            
        if row == len(grid) - 1 and col == len(grid[0]) - 1:
            return set([grid[row][col]])
            
        if (row, col) in memo:
            return memo[(row, col)]
            
        down = self.helper(row + 1, col, grid, memo)
        right = self.helper(row, col + 1, grid, memo)
        
        result = set([(obj + grid[row][col]) for obj in down.union(right)])
        
        memo[(row, col)] = result
        return result
    
    
    # Method.2      DP + 滚动数组 
    def uniqueWeightedPaths(self, grid):
        if not grid or not grid[0]:
            return 0
            
        dp = [set() for i in range(len(grid[0]))]
        dp[len(grid[0]) - 1].add(0)
        
        for row in range(len(grid) - 1, -1, -1):
            for col in range(len(grid[0]) - 1, -1, -1):
                if col + 1 < len(grid[0]):
                    dp[col] = dp[col].union(dp[col + 1])
                dp[col] = set([(obj + grid[row][col]) for obj in dp[col]])
                
        return sum(dp[0])





# 1543 · Unique Path IV
"""

"""
class Solution:
    """
    @param height: the given height
    @param width: the given width
    @return: the number of paths you can reach the end
    """
    # Method.1      动态规划，初始化左上到右下对角线，因为走法只有1种，然后每次斜着动态规划
    def uniquePath(self, height, width):
        dp = [[0 for j in range(width)] for i in range(height)]
        dx = [-1, 0, 1]
        dy = [-1, -1, -1]

        for i in range(height):
            if i < width:
                dp[i][i] = 1

        for i in range(1, width):
            for j in range(0, height):
                row = j
                col = j + i
                if col < width:
                    for k in range(3):
                        row_ = row + dx[k]
                        col_ = col + dy[k]
                        if 0 <= row_ < height and 0 <= col_ < width:
                            dp[row][col] += dp[row_][col_]
                else:
                    break

        return dp[0][width - 1] % 1000000007


    # Method.2
    def uniquePath(self, height, width):
        dp = [[0 for i in range(width)] for j in range(height)]
        dp[0][0] = 1

        for j in range(1, width):
            dp[0][j] = dp[0][j-1] + dp[1][j-1]
            for i in range(1, height-1):
                dp[i][j] = dp[i-1][j-1] + dp[i][j-1] + dp[i+1][j-1]
            dp[height-1][j] = dp[height-2][j-1] + dp[height-1][j-1]

        return dp[0][width-1] % 1000000007





# 795 · 4-Way Unique Paths
"""
A robot is located at the top-left corner of a m x n grid.
The robot can move any direction at any point in time, but every grid can only be up to once. 
The robot is trying to reach the bottom-right corner of the grid.
How many possible unique paths are there?
Input:  2 3
Output: 4
"""
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    """
    @param m: the row
    @param n: the column
    @return: the possible unique paths
    """
    def unique_paths(self, m: int, n: int) -> int:
        self.result =  0
        visited = [[0] * m for _ in range(n)]
        visited[0][0] = 1

        self.dfs(0, 0, m, n, visited)
        return self.result

    def dfs(self, x, y, m, n, visited):
        if x == n - 1 and y == m - 1:
            self.result += 1
            return

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy
            if not (0 <= next_x < n and 0 <= next_y < m):
                continue    
            if visited[next_x][next_y]:
                continue

            visited[next_x][next_y] = 1
            self.dfs(next_x, next_y, m, n, visited)
            visited[next_x][next_y] = 0