# 433/434 · Number of Islands I&II



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