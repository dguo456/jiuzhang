# 33 · N-Queens
"""
The N-queens puzzle is the problem of placing n queens on an n x n chessboard, and the queens can not be:
(Any two queens can't be in the same row, column, diagonal line). Given an integer n, return all distinct 
solutions to the N-queens puzzle. Each solution contains a distinct board configuration of the N-queens' placement, 
where 'Q' and '.' each indicate a queen and an empty space respectively.
Input:      n = 4
Output:
[
  // Solution 1
  [".Q..",
   "...Q",
   "Q...",
   "..Q."
  ],
  // Solution 2
  ["..Q.",
   "Q...",
   "...Q",
   ".Q.."
  ]
]
"""
class Solution:
    """
    @param: n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n):
        self.results = []
        self.dfs(n, [])
        return self.results
        
    def dfs(self, n, chessboard_cols):
        # row = len(chessboard_cols)
        if len(chessboard_cols) == n:
            self.results.append(self.draw_chessboard(chessboard_cols))
            return
        
        for col in range(n):
            if not self.isValid(chessboard_cols, col):
                continue
            chessboard_cols.append(col)
            self.dfs(n, chessboard_cols)
            chessboard_cols.pop()
            
    def draw_chessboard(self, chessboard_cols):
        n = len(chessboard_cols)
        board = []
        for i in range(n):
            row = ['Q' if j == chessboard_cols[i] else '.' for j in range(n)]
            board.append(''.join(row))
        return board
        
    def isValid(self, chessboard_cols, col):
        row = len(chessboard_cols)
        for r, c in enumerate(chessboard_cols):
            if c == col:
                return False
            if r-c == row-col or r+c == row+col:
                return False
        return True





# 34 · N-Queens II
"""
According to N-Queens problem. Now instead outputting board configurations, 
return the total number of distinct solutions.
Input:      n = 4
Output:     2
"""
class Solution:
    """
    @param n: The number of queens.
    @return: The total number of distinct solutions.
    """
    def totalNQueens(self, n):
        self.results = 0
        self.dfs(n, [])
        return self.results

    def dfs(self, n, cols):
        if len(cols) == n:
            # self.results.append(self.draw_chessboard(cols))
            self.results += 1
            return

        for col in range(n):
            if not self.is_valid(cols, col):
                continue
            cols.append(col)
            self.dfs(n, cols)
            cols.pop()

    # def draw_chessboard(self, cols):
    #     n = len(cols)
    #     board = []
    #     for i in range(n):
    #         row = ['Q' if j == cols[i] else '.' for j in range(n)]
    #         board.append(''.join(row))
    #     return board

    def is_valid(self, cols, col):
        row = len(cols)
        for r, c in enumerate(cols):
            if c == col:
                return False
            if r-c == row-col or r+c == row+col:
                return False
        return True