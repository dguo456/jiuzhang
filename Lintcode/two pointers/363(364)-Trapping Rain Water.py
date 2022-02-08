# 363 · Trapping Rain Water
"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it is able to trap after raining.

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
"""

# Method.1          双指针
class Solution:
    """
    @param heights: a list of integers
    @return: a integer
    """
    def trapRainWater(self, heights):
        
        if not heights:
            return 0
            
        left, right = 0, len(heights) - 1
        left_max, right_max = heights[left], heights[right]
        water = 0

        # 这里注意要left<=right，否则会报错
        while left <= right:
            if left_max < right_max:
                left_max = max(left_max, heights[left])
                water += left_max - heights[left]
                left += 1
            else:
                right_max = max(right_max, heights[right])
                water += right_max - heights[right]
                right -= 1
                
        return water



# Method.2          stack
class Solution:
    """
    @param heights: a list of integers
    @return: a integer
    """
    def trapRainWater(self, height):

        stack = []
        n = len(height)
        result = 0
        
        for i in range(n):
            
            while (len(stack) != 0) and (height[stack[-1]] < height[i]):
                
                pop_height = height[stack[-1]]
                stack.pop()
                
                # If the stack does not have any bars or the the popped bar has no left boundary
                if(len(stack) == 0):
                    break
            
                distance = i - stack[-1] - 1
                
                min_height = min(height[stack[-1]], height[i]) - pop_height
                
                result += distance * min_height
        
            stack.append(i)
        
        return result





# 364 · Trapping Rain Water II (Hard)
"""
将矩阵周边的格子都放到堆里，这些格子上面是无法盛水的。
每次在堆里挑出一个高度最小的格子 cell，把周围的格子加入到堆里。
这些格子被加入堆的时候，计算他们上面的盛水量。
盛水量 = cell.height - 这个格子的高度
当然如果这个值是负数，盛水量就等于 0。
"""
import heapq

class Solution:
    """
    @param heights: a matrix of integers
    @return: an integer
    """
    def trapRainWater(self, heights):
        if not heights:
            return 0
    
        self.initialize(heights)
        
        water = 0
        while self.borders:
            height, x, y = heapq.heappop(self.borders)
            for x_, y_ in self.adjcent(x, y):
                water += max(0, height - heights[x_][y_])
                self.add(x_, y_, max(height, heights[x_][y_]))

        return water

    def initialize(self, heights):
        self.n = len(heights)
        self.m = len(heights[0])
        self.visited = set()
        self.borders = []
        
        for index in range(self.n):
            self.add(index, 0, heights[index][0])
            self.add(index, self.m - 1, heights[index][self.m - 1])
            
        for index in range(self.m):
            self.add(0, index, heights[0][index])
            self.add(self.n - 1, index, heights[self.n - 1][index])
            
    def add(self, x, y, height):
        # add x, y, height to borders
        heapq.heappush(self.borders, (height, x, y))
        self.visited.add((x, y))
        
    def adjcent(self, x, y):
        adj = []
        for delta_x, delta_y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x_ = x + delta_x
            y_ = y + delta_y
            if 0 <= x_ < self.n and 0 <= y_ < self.m and (x_, y_) not in self.visited:
                adj.append((x_, y_))
        return adj