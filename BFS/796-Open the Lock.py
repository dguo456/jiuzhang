# 796 · Open the Lock
"""
You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', 
'5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', 
or '0' to be '9'. Each move consists of turning one wheel one slot.
The lock initially starts at '0000', a string representing the state of the 4 wheels.
You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels 
of the lock will stop turning and you will be unable to open it.
Given a target representing the value of the wheels that will unlock the lock, return the minimum total number 
of turns required to open the lock, or -1 if it is impossible.

Given deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Return 6

Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".
"""
from collections import deque 

class Solution:
    """
    @param deadends: the list of deadends
    @param target: the value of the wheels that will unlock the lock
    @return: the minimum total number of turns 
    """
    # 难点在于这是一个Implicit Graph，将其转化为：简单无向图知道起点和终点求最短路径问题，
    # 所以使用BFS解题，思路和Word Ladder非常类似，同 611
    def openLock(self, deadends, target):
        deadends_set = set(deadends)
        start = "0000"

        if start in deadends_set:
            return -1 

        queue = deque([start])
        distance = {"0000": 0}

        while queue:
            len_q = len(queue)
            
            for _ in range(len_q):
                code = queue.popleft()
                
                if code == target:
                    return distance[code]
                
                for next_code in self.get_next_codes(code, deadends_set):
                    if next_code in distance:
                        continue
                    queue.append(next_code)
                    distance[next_code] = distance[code] + 1 
                    
        return -1 
        
        
    def get_next_codes(self, code, deadends):
        next_codes = [] 
        
        for i in range(len(code)):
            left, mid, right = code[:i], code[i], code[i + 1:]
            for digit in [(int(mid) + 1) % 10, (int(mid) - 1) % 10]:
                next_code = left + str(digit) + right 
                
                if next_code in deadends:
                    continue 
                
                next_codes.append(next_code)
                
        return next_codes