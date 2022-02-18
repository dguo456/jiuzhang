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