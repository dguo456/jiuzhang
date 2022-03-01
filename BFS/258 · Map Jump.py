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
import sys
import heapq
from collections import deque
DIRECTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]

class Solution:
    """
    @param arr: the map
    @return:  the smallest target that satisfies from the upper left corner (0, 0) to the lower right corner (n-1, n-1)
    """
    # Method.1      SPFA的写法，用priority queue代替queue来加速计算，用一个hashmap记录到达(x,y)位置所需要的最小jump height，
    #               一个新的点可以进入队列 - 如果我们没有遇到过这种组合或者新组合的jump height更小
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