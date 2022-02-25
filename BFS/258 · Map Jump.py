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
import heapq

DIRECTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]

class Solution:
    """
    @param arr: the map
    @return:  the smallest target that satisfies from the upper left corner (0, 0) to the lower right corner (n-1, n-1)
    """
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
