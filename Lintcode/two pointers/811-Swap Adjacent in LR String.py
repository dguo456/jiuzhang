# 811 · Swap Adjacent in LR String
"""
In a string composed of 'L', 'R', and 'X' characters, like "RXXLRXRXL", a move consists of 
either replacing one occurrence of "XL" with "LX", or replacing one occurrence of "RX" with "XR". 
Given the starting string and the ending string, return True if and only if there exists a sequence 
of moves to transform one string to the other.

Input: start = `"RXXLRXRXL"`, end = `"XRLXXRRLX"`, 
Output: true
Explanation:
We can transform start to end following these steps:
RXXLRXRXL -> XRXLRXRXL -> XRLXRXRXL -> XRLXXRRXL -> XRLXXRRLX
"""

# Method.1      two pointer
class Solution:
    """
    @param start: the start
    @param end: the end
    @return: is there exists a sequence of moves to transform one string to the other
    """
    # L只能向左移动，R只能向右移动，因此对应位置的L，start中的L只能在end中L的右边（或者同一位置），
    # 对应位置的R，start中的R只能在end中的R的左边
    def canTransform(self, start, end):
        # Write your code here
        if start.replace('X','') != end.replace('X', ''):
            return False
        l = 0
        r = 0
        for i in range(len(start)):
            if start[i] == 'R':
                r += 1
            if end[i] == 'L':
                l += 1
            if start[i] == 'L':
                l -= 1
            if end[i] == 'R':
                r -= 1
            if (l < 0 or r != 0) and (l != 0 or r < 0):
                return False
        if l == 0 and r == 0:
            return True
        return False




# Method. 2
class Solution:
    """
    @param start: the start
    @param end: the end
    @return: is there exists a sequence of moves to transform one string to the other
    """
    def canTransform(self, start: str, end: str) -> bool:
        d = {
            'RX': 'XR',
            'XL': 'LX',
            'RXL': ['XRL', 'RLX']
        }

        if len(start) != len(end):
            return False

        index = 0
        while index < len(start):
            if start[index] != end[index]:
                if index + 1 < len(start) and start[index: index+2] in d:
                    if d[start[index: index+2]] == end[index: index+2]:
                        index += 2
                    else:
                        return False
                elif index + 2 < len(start) and start[index: index+3] in d:
                    if end[index: index+3] in d[start[index: index+3]]:
                        index += 3
                    else:
                        return False

                else:
                    return False

            else:
                index += 1

        return True