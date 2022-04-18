# 1855 · Reach Destination
"""
Given a starting point (sx, sy) and an ending point (dx, dy). If the starting point can be converted to 
the ending point through a series of conversions, return true, otherwise return false.
The conversion rule is: you can transform point (x, y) to (x, x + y) or (x + y, y).

Input: 
sx = 1, sy = 1, dx = 3, dy = 5
Output: 
True
Explanation:
You can do the following transformations
(1, 1) -> (1, 2)
(1, 2) -> (3, 2)
(3, 2) -> (3, 5)
"""
from collections import deque

class Solution:
    """
    @param sx: the start x
    @param sy: the start y
    @param dx: the destination x
    @param dy: the destination y
    @return: whether you can reach the destination
    """
    # Method.1      BFS，从起点走向终点，会超时
    def reach_destination(self, sx: int, sy: int, dx: int, dy: int) -> bool:
        if dx < sx or dy < sy:
            return False

        queue = deque([(sx, sy)])
        destination = (dx, dy)

        while queue:
            x, y = queue.popleft()

            if (x, y) == destination:
                return True
            # 剪枝操作，若x相等时，则提前判断是否y多余的值可以被x整除。
            if x == dx and (dy - y) % x == 0:
                return True
            # 剪枝操作，若y相等时，则提前判断是否x多余的值可以被y整除。
            if y == dy and (dx - x) % y == 0:
                return True

            if x + y <= dx:
                queue.append((x+y, y))
            if x + y <= dy:
                queue.append((x, x+y))

        return False


    # Method.2      由起点走到终点其实等同于从终点走向起点。每次转换加法操作则变为了减法操作，
    #               所以，剪枝条件为当前的currX和当前的currY必须大于sx和sy。首先，判断dx和dy
    #               是否都大于sx和sy，若不满足则直接输出False；然后使用队列将可行点坐标(currX,currY)
    #               压入队列中，直到currX和currY等于sx和sy，或小于sx和sy，则跳出循环。
    def reach_destination(self, sx: int, sy: int, dx: int, dy: int) -> bool:
        if dx < sx or dy < sy:
            return False
        
        queue = deque()
        queue.append([dx, dy])
        while queue:
            current = queue.popleft()
            x, y = current[0], current[1]
            if x == sx and y == sy:
                return True
            # 剪枝操作，若x相等时，则提前判断是否y多余的值可以被x整除。
            if x == sx and y % x == 0:
                return True
            # 剪枝操作，若y相等时，则提前判断是否x多余的值可以被y整除。
            if y == sy and x % y == 0:
                return True
            if x > y:
                if x - y >= sx:
                    queue.append([x - y, y])
            else:
                if y - x >= sy:
                    queue.append([x, y - x])
                    
        return False