# 433/434 - Number of Islands I&II
# 1080    - Max Area of Island
# 574/573 - Build Post Office I&II
# 477     - Surrounded Regions
# 663     - Walls and Gates
# 778     - Pacific Atlantic Water Flow
# 1367    - Police Distance
# 1563    - Shortest path to the destination
# 1225    - Island Perimeter
################################################################################################################



# 433   Number of Islands
from collections import defaultdict, deque

DIRECTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

# 433 · Number of Islands
"""
Given a boolean 2D matrix, 0 is represented as the sea, 1 is represented as the island. 
If two 1 is adjacent, we consider them in the same island. We only consider up/down/left/right adjacent.
Find the number of islands.
"""
class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        if not grid or len(grid[0]) == 0:
            return 0

        visited = set()
        res = 0

        # 不是最短路径问题，因此需要遍历整个 Matrix 找一共有几个岛屿
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] and (i, j) not in visited:
                    self.bfs(grid, i, j, visited)
                    res += 1

        return res

    def bfs(self, grid, x, y, visited):
        queue = deque([(x, y)])
        visited.add((x, y))

        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                nextX, nextY = x + dx, y + dy
                if not self.isValid(grid, nextX, nextY, visited):
                    continue
                queue.append((nextX, nextY))
                visited.add((nextX, nextY))

    def isValid(self, grid, x, y, visited):
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
            return False
        if (x, y) in visited:
            return False
        # 因为是boolean类型，return值为true或者false，如果是不是岛屿（false）就continue
        return grid[x][y]




# 434 · Number of Islands II
"""
Given a n,m which means the row and column of the 2D matrix and an array of pair A (size k). 
Originally, the 2D matrix is all 0 which means there is only sea in the matrix. The list pair 
has k operator and each operator has two integer A[i].x, A[i].y means that you can change the 
grid matrix[A[i].x][A[i].y] from sea to island. Return how many island are there in the matrix 
after each operator.You need to return an array of size K.
"""

# Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

class Solution:
    """
    @param n: An integer
    @param m: An integer
    @param operators: an array of point
    @return: an integer array
    """
    def numIslands2(self, n, m, operators):
        if not operators:
            return []
        
        islands = set()
        self.size = 0
        self.parent = {}
        results = []
        
        for operator in operators:
            x, y = operator.x, operator.y
            if (x, y) in islands:
                results.append(self.size)
                continue
            
            islands.add((x, y))
            self.size += 1
            self.parent[(x, y)] = (x, y)
            for delta_x, delta_y in DIRECTIONS:
                next_x = x + delta_x
                next_y = y + delta_y
                if (next_x, next_y) in islands:
                    self.union((next_x, next_y), (x, y))
            
            results.append(self.size)

        return results
        
    def union(self, pointA, pointB):
        rootA = self.find(pointA)
        rootB = self.find(pointB)
        if rootA != rootB:
            self.parent[rootA] = rootB
            self.size -= 1
            
    def find(self, point):
        path = []
        while point != self.parent[point]:
            path.append(point)
            point = self.parent[point]
            
        for p in path:
            self.parent[p] = point
        
        return point






# 1080 · Max Area of Island
"""
Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 
4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)

input:
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
output : 6.
Explanation : Note the answer is not 11, because the island must be connected 4-directionally.
"""
# 不难，但是要注意更新grid为0的顺序，尤其是在(next_x, next_y)下面而不是在(x, y)下面，值得多想想
class Solution:
    """
    @param grid: a 2D array
    @return: the maximum area of an island in the given 2D array
    """
    def max_area_of_island(self, grid) -> int:
        if not grid or not grid[0]:
            return 0

        max_area = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    grid[r][c] = 0
                    area = self.bfs(grid, r, c)
                    max_area = max(area, max_area)

        return max_area

    def bfs(self, grid, x, y):
        queue = deque([(x, y)])
        area = 1

        while queue:
            x, y = queue.popleft()

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy

                if not (0 <= next_x < len(grid) and 0 <= next_y < len(grid[0])):
                    continue
                if grid[next_x][next_y] == 0:
                    continue
                
                queue.append((next_x, next_y))
                grid[next_x][next_y] = 0
                area += 1

        return area






# 477 · Surrounded Regions
"""
Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
A region is captured by flipping all 'O''s into 'X''s in that surrounded region.
Input:
  X X X X
  X O O X
  X X O X
  X O X X
Output:
  X X X X
  X X X X
  X X X X
  X O X X
"""
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class Solution:
    """
    @param: board: board a 2D board containing 'X' and 'O'
    @return: nothing
    """
    def surroundedRegions(self, board):
        if not board or not board[0]:
            return []

        visited = set()
        surrounded_regions = []

        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == "O" and (r, c) not in visited and \
                (r == 0 or r == len(board)-1 or c == 0 or c == len(board[0])-1):
                    self.bfs(board, visited, r, c)

        # 边界上的O点肯定不会被包围，所以以所有在边界的O点作为起点做BFS并一个个标记成F。然后，凡是没有被标记过的O点都变成X，就得出答案了。
        visited = set()
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "F":
                    board[r][c] = "O"

        

    def bfs(self, board, visited, x, y):
        queue = deque([(x, y)])
        visited.add((x, y))
        board[x][y] = "F"

        while queue:
            x, y = queue.popleft()

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy

                if not (0 <= next_x < len(board) and 0 <= next_y < len(board[0])):
                    continue
                if (next_x, next_y) in visited:
                    continue
                if board[next_x][next_y] == "O":
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))
                    board[next_x][next_y] = "F"






# 574 · Build Post Office
"""
Given a 2D grid, each cell is either an house 1 or empty 0 (It is represented by the numbers 0, 1), 
find the place to build a post office, the distance that post office to all the house sum is smallest. 
Returns the sum of the minimum distances from all houses to the post office, Return -1 if it is not possible.
1.  You can pass through house and empty.
2.  You only build post office on an empty.
3.  The distance between house and the post office is Manhattan distance

Input:  [[0,1,0,0],[1,0,1,1],[0,1,0,0]]
Output:  6
Explanation:    Placing a post office at (1,1), the distance that post office to all the house sum is smallest.
"""
import sys

class Solution:
    """
    @param grid: a 2D grid
    @return: An integer
    """
    # Method.1      普通simulation方法， 会超时
    def shortestDistance(self, grid):
        if not grid or len(grid) == 0 or len(grid[0]) == 0:
            return -1

        result = sys.maxsize
        distance = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    distance.append((r, c))

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 0:
                    result = min(self.get_distance(distance, r, c), result)

        return result

    def get_distance(self, distance, x, y):
        min_distance = 0

        for dx, dy in distance:
            min_distance += abs(x - dx) + abs(y - dy)

        return min_distance



    # Method.2      nb改进方法
    # 注意本题与post office II 不同：house是可以穿越的，所以BFS不是最优解，直接使用数学计算求解。
    # （1）把house的坐标分别投影到x，y轴上。以x轴为例，用数组 row_count 记录所有具有相同x坐标值的house数量。
    # （2）对于任一empty位置的x坐标值i，它与所有house在x轴上投影点的距离和，用数组row_distance记录。在计算row_distance[i]时，
    #       遍历所有的x坐标。对于其中每一个坐标j，它的累加距离为i与j的距离，乘以该点（j）上拥有的house总量，
    #       即  row_distance[i] += row_count[j] * abs(j - i)。对于y坐标进行类似的处理。
    def shortestDistance(self, grid):
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        row_count, col_count = [0] * m, [0] * n
        result = sys.maxsize
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_count[i] += 1
                    col_count[j] += 1

        row_distance = self.get_distance(row_count)
        col_distance = self.get_distance(col_count)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    result = min(result, row_distance[i] + col_distance[j])
        return result

    def get_distance(self, count):
        length = len(count)
        distance = [0] * length
        for i in range(length):
            for j in range(length):
                distance[i] += count[j] * abs(j - i)
        return distance






# 573 · Build Post Office II
"""
Given a 2D grid, each cell is either a wall 2, an house 1 or empty 0 (the number zero, one, two), find a place 
to build a post office so that the sum of the distance from the post office to all the houses is smallest.
Returns the sum of the minimum distances from all houses to the post office.Return -1 if it is not possible.
You cannot pass through wall and house, but can pass through empty.
You only build post office on an empty.

Input:  [[0,1,0,0,0],[1,0,0,2,1],[0,1,0,0,0]]
Output: 8
Explanation:    Placing a post office at (1,1), the distance that post office to all the house sum is smallest.
"""
from collections import deque

DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    """
    @param grid: a 2D grid
    @return: An integer
    """
    # 此题难点在于需要逆向思维, 从house开始BFS, 枚举邮局(1)位置, 否则从empty开始会TLE
    # 这个题告诉我们 当TLE时候 去集合中找找其他部分数量少的 来进行操作 尤其对这种单一操作BFS时间消耗大的题目
    def shortestDistance(self, grid):
        if not grid or len(grid) == 0 or len(grid[0]) == 0:
            return -1

        m = len(grid)
        n = len(grid[0])
        
        distance = [[sys.maxsize for _ in range(n)] for _ in range(m)]
        reachable_count = [[0 for _ in range(n)] for _ in range(m)]
        min_dist = sys.maxsize
        
        buildings = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    self.bfs(grid, i, j, distance, reachable_count)
                    buildings += 1
  
        for i in range(m):
            for j in range(n):
                # 此处判断从该点出发能不能reach到所有的邮局，能的话就打擂台更新最小距离
                if reachable_count[i][j] == buildings and distance[i][j] < min_dist:
                    min_dist = distance[i][j]

        return min_dist if min_dist != sys.maxsize else -1
        
    def bfs(self, grid, r, c, distance, reachable_count):
        queue = deque([(r, c)])
        visited = set()
        visited.add((r, c))
        level = 0
        
        while queue:

            for _ in range(len(queue)):
                x, y = queue.popleft()
                # 如果当前点 distance 没有被访问到则初始化为0
                if distance[x][y] == sys.maxsize:
                    distance[x][y] = 0
                distance[x][y] += level

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not self.is_valid(grid, visited, next_x, next_y):
                        continue
                    if grid[next_x][next_y] == 0:
                        queue.append((next_x, next_y))
                        reachable_count[next_x][next_y] += 1
                    visited.add((next_x, next_y))
            
            level += 1

    def is_valid(self, grid, visited, x, y):
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
            return False
        if (x, y) in visited:
            return False
        return True






# 1367 · Police Distance
"""
Given a matrix size of n x m, element 1 represents policeman, -1 represents wall and 0 represents empty.
Now please output a matrix size of n x m, output the minimum distance between each empty space and the 
nearest policeman

Input: 
mat =
[
    [0, -1, 0],
    [0, 1, 1],
    [0, 0, 0]
]
Output: [[2,-1,1],[1,0,0],[2,1,1]]
Explanation:
The distance between the policeman and himself is 0, the shortest distance between the two policemen to 
other empty space is as shown above
"""
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    """
    @param matrix : the martix
    @return: the distance of grid to the police
    """
    def police_distance(self, matrix):
        if not matrix or not matrix[0]:
            return []

        queue = deque([])
        police_office = set()

        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if matrix[r][c] == 1:
                    queue.append((r, c))
                    police_office.add((r, c))

        level = 0
        while queue:
            level += 1
            len_queue = len(queue)

            for _ in range(len_queue):
                x, y = queue.popleft()

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy

                    if not (0 <= next_x < len(matrix) and 0 <= next_y < len(matrix[0])):
                        continue
                    if matrix[next_x][next_y] == -1:
                        continue
                    if (next_x, next_y) in police_office:
                        continue
                    
                    if matrix[next_x][next_y] == 0 or level < matrix[next_x][next_y]:
                        queue.append((next_x, next_y))
                        matrix[next_x][next_y] = level
        
        for x, y in police_office:
            matrix[x][y] = 0
            
        return matrix






# 663 · Walls and Gates
"""
You are given a m x n 2D grid initialized with these three possible values.
-1 - A wall or an obstacle.
0 - A gate.
INF - Infinity means an empty room. We use the value 2^31 - 1 = 2147483647 to represent INF as you may assume 
        that the distance to a gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a Gate, 
that room should remain filled with INF
"""
# Input:
# [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],
#   [2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
# Output:
# [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]

# Explanation:
# the 2D grid is:
# INF  -1  0  INF
# INF INF INF  -1
# INF  -1 INF  -1
#   0  -1 INF INF
# the answer is:
#   3  -1   0   1
#   2   2   1  -1
#   1  -1   2  -1
#   0  -1   3   4
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    """
    @param rooms: m x n 2D grid
    @return: nothing
    """
    def walls_and_gates(self, rooms):
        if not rooms or not rooms[0]:
            return -1

        queue = deque()

        for r in range(len(rooms)):
            for c in range(len(rooms[0])):
                if rooms[r][c] == 0:
                    queue.append((r, c))

        distance = 0
        while queue:
            distance += 1
            len_q = len(queue)

            for _ in range(len_q):
                x, y = queue.popleft()

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy

                    if not (0 <= next_x < len(rooms) and 0 <= next_y < len(rooms[0])):
                        continue

                    if rooms[next_x][next_y] == -1:
                        continue

                    if distance < rooms[next_x][next_y]:
                        queue.append((next_x, next_y))
                        rooms[next_x][next_y] = distance

# Method.2      从所有INF出发寻找0，会超时
class Solution:
    """
    @param rooms: m x n 2D grid
    @return: nothing
    """
    def walls_and_gates(self, rooms):
        if not rooms or not rooms[0]:
            return -1
        
        for r in range(len(rooms)):
            for c in range(len(rooms[0])):
                if rooms[r][c] == 2147483647:
                    self.bfs(rooms, r, c)

        return rooms
    
    def bfs(self, rooms, start_x, start_y):
        queue = deque([(start_x, start_y, 0)])
        visited = set([(start_x, start_y)])

        while queue:
            x, y, level = queue.popleft()

            if rooms[x][y] == 0:
                rooms[start_x][start_y] = level
                return

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy
                if not self.is_valid(rooms, visited, next_x, next_y):
                    continue
                queue.append((next_x, next_y, level+1))
                visited.add((next_x, next_y))

    def is_valid(self, rooms, visited, x, y):
        if x < 0 or x >= len(rooms) or y < 0 or y >= len(rooms[0]):
            return False
        if (x, y) in visited:
            return False
        if rooms[x][y] == -1:
            return False
        return True






# 778 · Pacific Atlantic Water Flow
"""
Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, 
the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right 
and bottom edges. Water can only flow in four directions (up, down, left, or right) from a cell to another one 
with height equal or lower. Find the list of grid coordinates where water can flow to both the Pacific and 
Atlantic ocean.

Input:
matrix = 
[[1,2,2,3,5],
[3,2,3,4,4],
[2,4,5,3,1],
[6,7,1,4,5],
[5,1,1,2,4]]
Output:
[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Explanation:
Pacific ~ ~ ~ ~ ~
      ~ 1 2 2 3 5 *
      ~ 3 2 3 4 4 *
      ~ 2 4 5 3 1 *
      ~ 6 7 1 4 5 *
      ~ 5 1 1 2 4 *
        * * * * * Atlantic
"""
class Solution:
    """
    @param matrix: the given matrix
    @return: The list of grid coordinates
    """
    # Method.1      BFS - 从每一个点出发做bfs，能过但是需要优化
    def pacific_atlantic(self, matrix):
        if not matrix or not matrix[0]:
            return []

        results = []
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if self.bfs(matrix, r, c):
                    results.append([r, c])

        return results

    def bfs(self, matrix, x, y):
        queue = deque([(x, y)])
        visited = set([(x, y)])
        pacific = atlantic = False

        while queue:
            x, y = queue.popleft()

            if x == 0 or y == 0:
                pacific = True
            if x == len(matrix) - 1 or y == len(matrix[0]) - 1:
                atlantic = True

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy

                if not (0 <= next_x < len(matrix) and 0 <= next_y < len(matrix[0])):
                    continue
                if (next_x, next_y) in visited:
                    continue
                if matrix[next_x][next_y] > matrix[x][y]:
                    continue

                queue.append((next_x, next_y))
                visited.add((next_x, next_y))

        return pacific and atlantic


    # Method.2      优化方法，BFS边缘向内部反推
    def pacific_atlantic(self, matrix):
        if not matrix or not matrix[0]:
            return []

        n, m = len(matrix), len(matrix[0])
        pacific_queue, atlantic_queue = deque(), deque()
        pacific_visited, atlantic_visited = set(), set()

        for r in range(n):
            pacific_queue.append((r, 0))
            pacific_visited.add((r, 0))
            atlantic_queue.append((r, m - 1))
            atlantic_visited.add((r, m - 1))

        for c in range(m):
            pacific_queue.append((0, c))
            pacific_visited.add((0, c))
            atlantic_queue.append((n - 1, c))
            atlantic_visited.add((n - 1, c))

        self.bfs(matrix, pacific_queue, pacific_visited)
        self.bfs(matrix, atlantic_queue, atlantic_visited)
        
        return [visited for visited in pacific_visited.intersection(atlantic_visited)]

    def bfs(self, matrix, queue, visited):

        while queue:
            x, y = queue.popleft()

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy

                if not (0 <= next_x < len(matrix) and 0 <= next_y < len(matrix[0])):
                    continue
                if (next_x, next_y) in visited:
                    continue
                if matrix[next_x][next_y] < matrix[x][y]:
                    continue

                queue.append((next_x, next_y))
                visited.add((next_x, next_y))







# 1563 · Shortest path to the destination
"""
Given a 2D array representing the coordinates on the map, there are only values 0, 1, 2 on the map. 
value 0 means that it can pass, value 1 means not passable, value 2 means target place. 
Starting from the coordinates [0,0],You can only go up, down, left and right. 
Find the shortest path that can reach the destination, and return the length of the path.
"""

DIRECTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

from collections import deque

class Solution:
    """
    @param targetMap: 
    @return: nothing
    """
    def shortestPath(self, targetMap):
        
        queue = deque([(0, 0)])
        visited = set()
        visited.add((0, 0))
        path = 0

        while queue:

            for _ in range(len(queue)):
                x, y = queue.popleft()

                if targetMap[x][y] == 2:
                    return path

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not self.is_valid(targetMap, visited, next_x, next_y):
                        continue
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))

            path += 1

        return -1

    def is_valid(self, targetMap, visited, x, y):
        if x < 0 or x >= len(targetMap) or y < 0 or y >= len(targetMap[0]):
            return False
        if (x, y) in visited:
            return False
        if targetMap[x][y] == 1:
            return False
        return True






# 1225 · Island Perimeter
"""
You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water. 
Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, 
and there is exactly one island (i.e., one or more connected land cells). The island doesn't have "lakes" 
(water inside that isn't connected to the water around the island). One cell is a square with side length 1. 
The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
Answer: 16
"""
class Solution:
    """
    @param grid: a 2D array
    @return: the perimeter of the island
    """
    def island_perimeter(self, grid) -> int:
        if not grid or not grid[0]:
            return -1

        result = 0
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    result += self.count(grid, r, c)

        return result

    def count(self, grid, x, y):
        count = 0
        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy

            if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
                count += 1
            elif grid[next_x][next_y] == 0:
                count += 1

        return count