# 569 · Add Digits
"""
Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
Input:  num=38
Output: 2
Explanation:
The process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return 2.
"""

# 1384 = 4 + 8*10 + 3*100 + 1*1000 = (4+8+3+1) + 9*(8+3+1) + 90*(3+1) + 900*1
# 这里我们只需要保留（4+8+3+1），其余的都是可以被9整除的项都可以去掉不考虑
class Solution:
    """
    @param num: a non-negative integer
    @return: one digit
    """
    def addDigits(self, num):
        if num == 0:
            return 0

        return (num - 1) % 9 + 1