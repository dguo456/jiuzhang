# 258  · Map Jump
# 1446 · 01 Matrix Walking Problem
# 1723 · Shortest Path in a Grid with Obstacles Elimination
# 1829 · Find the number of shortest path
# 1832 · Minimum Step
# 651  · Binary Tree Vertical Order Traversal
# 1862 · Time to Flower Tree
##################################################################################################################


import sys
import heapq
from collections import defaultdict, deque

DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# 258 · Map Jump
"""
Given a map of n*n, each cell has a height. Each time you can only move to an adjacent cell, 
and the height difference between the two cells is required to not exceed target. You cannot 
get out of the map. Find the the smallest target that satisfies from the upper left corner 
(0, 0) to the lower right corner (n-1, n-1).
Input:[[1,5],[6,2]],
Output:4,
Explanation:
There are 2 ways:
1. 1 -> 5 -> 2 The target in this way is 4.
2. 1 -> 6 -> 2 The target in this way is 5.
So the answer is 4.
"""
class Solution:
    """
    @param arr: the map
    @return:  the smallest target that satisfies from the upper left corner (0, 0) to the lower right corner (n-1, n-1)
    """
    # Method.1      SPFA的写法，用priority queue代替queue来加速计算，用一个hashmap记录到达(x,y)位置所需要的最小
    #               jump height，如何判断一个新的点可以进入队列 - 如果我们没有遇到过这种组合或者新组合的jump height更小
    def mapJump(self, arr):
        if not arr or not arr[0]:
            return 0

        m, n = len(arr), len(arr[0])
        queue = [(0, 0, 0)]
        visited = set([(0, 0, 0)])
        min_height = {(0, 0) : 0}

        while queue:
            x, y, height = heapq.heappop(queue)

            if x == m - 1 and y == n - 1:
                return height
            
            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy
                if not (0 <= next_x < m and 0 <= next_y < n):
                    continue
                
                next_height = max(height, abs(arr[next_x][next_y] - arr[x][y]))
                if (next_x, next_y, next_height) in visited:
                    continue
                if (next_x, next_y) in min_height and min_height[(next_x, next_y)] < next_height:
                    continue
                heapq.heappush(queue, (next_x, next_y, next_height))
                visited.add((next_x, next_y, next_height))
                min_height[(next_x, next_y)] = next_height



    # Method.2      二分target,看是否对于当前的target存在一条合法路径。时间复杂度 O(log（max(arr)）*n*m)，空间复杂度 O(n*m)。
    def map_jump(self, arr) -> int:
        if not arr or not arr[0]:
            return -1

        start, end = 0, -sys.maxsize
        for r in range(len(arr)):
            for c in range(len(arr[0])):
                end = max(arr[r][c], end)

        while start + 1 < end:
            mid = (start + end) // 2
            if self.can_reach_target(arr, mid):
                end = mid
            else:
                start = mid

        if self.can_reach_target(arr, start):
            return start
        if self.can_reach_target(arr, end):
            return end

        return -1

    def can_reach_target(self, arr, target):
        queue = deque([(0, 0)])
        visited = set((0, 0))

        while queue:
            x, y = queue.popleft()
            height = arr[x][y]

            if x == len(arr) - 1 and y == len(arr[0]) - 1:
                return True

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy
                if not (0 <= next_x < len(arr) and 0 <= next_y < len(arr[0])):
                    continue
                if (next_x, next_y) in visited:
                    continue
                if abs(height - arr[next_x][next_y]) > target:
                    continue
                
                queue.append((next_x, next_y))
                visited.add((next_x, next_y))

        return False





# 1446 · 01 Matrix Walking Problem
"""
Given an 01 matrix gird of size n*m, 1 is a wall, 0 is a road, now you can turn a 1 in the grid into 0, 
Is there a way to go from the upper left corner to the lower right corner? If there is a way to go, 
how many steps to take at least?

Input: a = [[0,1,0,0,0],[0,0,0,1,0],[1,1,0,1,0],[1,1,1,1,0]] 
Output:7 
Explanation:    Change `1` at (0,1) to `0`, the shortest path is as follows:
(0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4)->(3,4) There are many other options of length `7`, 
not listed here.
"""
class Solution:
    """
    @param grid: The gird
    @return: Return the steps you need at least
    """
    def get_best_road(self, grid) -> int:
        if not grid or not grid[0]:
            return -1

        start = (0, 0, grid[0][0])
        queue = deque([start])
        grid_map = [[float('inf') for i in range(len(grid[0]))] for j in range(len(grid))]

        distance = 0
        while queue:
            len_q = len(queue)

            for _ in range(len_q):
                x, y, wall = queue.popleft()

                if x == len(grid) - 1 and y == len(grid[0]) - 1:
                    return distance

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not (0 <= next_x < len(grid) and 0 <= next_y < len(grid[0])):
                        continue
                    
                    next_wall = wall + grid[next_x][next_y]
                    if next_wall > 1:
                        continue
                    if grid_map[next_x][next_y] <= next_wall:
                        continue

                    queue.append((next_x, next_y, next_wall))
                    grid_map[next_x][next_y] = min(next_wall, grid_map[next_x][next_y])

            distance += 1

        return -1







# 1723 · Shortest Path in a Grid with Obstacles Elimination    （同上题）
"""
Given a m * n grid, where each cell is either 0 (empty) or 1 (obstacle). In one step, you can move up, down, 
left or right from and to an empty cell. Return the minimum number of steps to walk from the upper left corner 
(0, 0) to the lower right corner (m-1, n-1) given that you can eliminate at most k obstacles. If it is not 
possible to find such walk return -1.

Input: 
grid = 
[[0,0,0],
 [1,1,0],
 [0,0,0],
 [0,1,1],
 [0,0,0]], 
k = 1
Output: 6
Explanation: 
The shortest path without eliminating any obstacle is 10. 
The shortest path with one obstacle elimination at position (3,2) is 6. Such path is 
(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> (4,2).
"""
class Solution:
    """
    @param grid: a list of list
    @param k: an integer
    @return: Return the minimum number of steps to walk
    """
    # Method.1      十分值得一做的题。就是普通的宽度优先搜索算法就行。唯一需要变化的点是，如果不可以移除障碍的话，
    #               宽度优先搜索的每个状态就是一个二维坐标(x,y)，放到队列里的和放到哈希表里的都是这个坐标。
    #               而现在有障碍的限制，那么我们将已经移除了多少个障碍 obs 放到状态里，形成一个三维的状态：
    #               (x, y, obs)，需要额外开一个二维数组记录每个点当前能够到达该点的最小障碍数
    def shortest_path(self, grid, k) -> int:
        if not grid or not grid[0]:
            return -1

        start = (0, 0, grid[0][0])
        queue = deque([start])
        min_k = [[float('inf') for i in range(len(grid[0]))] for j in range(len(grid))]

        distance = 0
        while queue:
            len_q = len(queue)

            for _ in range(len_q):
                x, y, obstacle = queue.popleft()

                if x == len(grid) - 1 and y == len(grid[0]) - 1:
                    return distance

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not (0 <= next_x < len(grid) and 0 <= next_y < len(grid[0])):
                        continue
                    
                    next_obstacle = obstacle + grid[next_x][next_y]
                    if next_obstacle > k:
                        continue
                    if min_k[next_x][next_y] <= next_obstacle:
                        continue

                    queue.append((next_x, next_y, next_obstacle))
                    min_k[next_x][next_y] = min(next_obstacle, min_k[next_x][next_y])

            distance += 1

        return -1


    # Method.2      DFS
    def __init__(self):
        self.n, self.m = 0, 0
        self.min_path_length = sys.maxsize
        self.x_dir = [1, 0, -1, 0]
        self.y_dir = [0, 1, 0, -1]

    def shortestPath(self, grid, k):
        count = 0
        self.m, self.n = len(grid), len(grid[0])
        
        self.dfs(grid, 0, 0, k, count)
        
        if self.min_path_length == sys.maxsize:
            return -1
        else:
            return self.min_path_length
    
    def dfs(self, grid, x, y, k, count):
        if not self.isValid(x, y) or grid[x][y] == -1:
            return
        
        if (k == 0 and grid[x][y] == 1) or self.min_path_length == self.m + self.n - 2:
            return
        
        if x == self.m - 1 and y == self.n - 1:
            if count < self.min_path_length:
                self.min_path_length = count
                return
        
        if grid[x][y] == 1:
            k -= 1
        
        temp = grid[x][y]
        grid[x][y] = -1
        for i in range(4):
            next_x, next_y = self.x_dir[i] + x, self.y_dir[i] + y
            self.dfs(grid, next_x, next_y, k, count + 1)
        grid[x][y] = temp

    def isValid(self, x, y):
        if 0 <= x < self.m and 0 <= y < self.n:
            return True
        return False






# 1829 · Find the number of shortest path
"""
Given a map of nn rows and mm columns, where 0 represents the road that can be taken and 1 represents the wall, 
you can start from any position in the first row to any position in the last row. You need return the number of 
shortest path. When either of the two paths is different, we call it a different path.
Input:      6   6
[[1,1,0,0,1,0],[1,1,0,0,1,1],[0,1,0,0,1,1],[0,1,1,0,0,1],[0,0,0,1,0,1],[0,0,0,1,0,1]]
Output:     1
Explanation: As is seen in the picture, the red road is the only way which length = 6
"""
class Solution:
    """
    @param n: the row of the map
    @param m: the column of the map
    @param labyrinth: the map
    @return: the number of shortest path
    """
    def the_numberof_shortest_path(self, n, m, labyrinth) -> int:
        if not labyrinth or not labyrinth[0] or n <= 0 or m <= 0:
            return -1

        shortest_length, total_path_nums = 0, 0
        temp_list = []
        for col in range(m):
            if labyrinth[0][col] == 0:
                exist_a_path, length, path_nums = self.bfs(n, m, labyrinth, col)
                if exist_a_path:
                    print((length, path_nums))
                    temp_list.append((length, path_nums))
        
        shortest_length = sorted(temp_list)[0][0]
        for x, y in temp_list:
            if x == shortest_length:
                total_path_nums += y

        return total_path_nums

    def bfs(self, n, m, labyrinth, col):
        queue = deque([(0, col)])
        visited = set((0, col))
        reachable = False
        distance, path_nums = 0, 0

        while queue:
            # 这里注意，需要每次更新visited，因为会有路径因为被访问过所以无法进queue
            temp_visited = visited.copy()
            len_q = len(queue)

            for i in range(len_q):
                x, y = queue.popleft()

                if x == n - 1:
                    reachable = True
                    path_nums = 1
                    for j in range(i+1, len_q):
                        temp_x, temp_y = queue.popleft()
                        if temp_x == n - 1:
                            path_nums += 1

                    return reachable, distance, path_nums

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not (0 <= next_x < n and 0 <= next_y < m):
                        continue
                    if (next_x, next_y) in visited:
                        continue
                    if labyrinth[next_x][next_y] == 1:
                        continue
                    queue.append((next_x, next_y))
                    temp_visited.add((next_x, next_y))
            
            distance += 1
            visited = temp_visited.copy()

        return False, 0, 0





# 1832 · Minimum Step
"""
There is a 1 * n chess table, indexed with 0, 1, 2 .. n - 1, every grid is colored.
And there is a chess piece on position 0, please calculate the minimum step that 
you should move it to position n-1.

Here are 3 ways to move the piece, the piece can't be moved outside of the table:
1.  Move the piece from position ii to position i + 1.
2.  Move the piece from position ii to positino i - 1.
3.  If the colors on position i and position j are same, you can 
    move the piece directly from position i to position j.
"""

# Input:
# colors = [1, 2, 3, 3, 2, 5]
# Output: 3
# Explanation: 
# In the example. you should move the piece 3 times:
# 1. Move from position 0 to position 1.
# 2. Because of the same color in position 1 and position 4, move from position 1 to position 4,
# 3. Move from position 4 to position 5.

class Solution:
    """
    @param colors: the colors of grids
    @return: return the minimum step from position 0 to position n - 1
    """
    def minimumStep(self, colors):
        if not colors or len(colors) == 0:
            return 0

        graph = defaultdict(list)
        for index, color in enumerate(colors):
            graph[color].append(index)

        queue = deque([0])
        visited = set()
        visited.add(0)
        steps = -1

        while queue:
            steps += 1

            for _ in range(len(queue)):
                index = queue.popleft()

                if index == len(colors) - 1:
                    return steps

                for next_index in graph[colors[index]]:
                    if next_index == index:
                        continue
                    if next_index not in visited:
                        queue.append(next_index)
                        visited.add(next_index)

                graph[colors[index]] = []

                if index + 1 < len(colors) and index+1 not in visited:
                    queue.append(index+1)
                    visited.add(index+1)

                if index - 1 >= 0 and index-1 not in visited:
                    queue.append(index-1)
                    visited.add(index-1)

        return steps





# 651 · Binary Tree Vertical Order Traversal
"""
Given a binary tree, return the vertical order traversal of its nodes' values. 
(ie, from top to bottom, column by column).
If two nodes are in the same row and column, the order should be from left to right.
Input: {3,9,8,4,0,1,7}
Output: [[4],[9],[3,0,1],[8],[7]]
Explanation:
     3
    /\
   /  \
   9   8
  /\  /\
 /  \/  \
 4  01   7
"""

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the root of tree
    @return: the vertical order traversal
    """
    def vertical_order(self, root: TreeNode):
        queue = deque()
        results = defaultdict(list)

        queue.append((root, 0))

        while queue:
            node, curr_level = queue.popleft()

            if node:
                results[curr_level].append(node.val)
                queue.append((node.left, curr_level - 1))
                queue.append((node.right, curr_level + 1))

        return [results[i] for i in sorted(results)]