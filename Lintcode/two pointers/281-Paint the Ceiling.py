# 281 · Paint the Ceiling
"""
Si = (k * Si-1 + b) % m + 1 + Si-1      for 1 <= i < n
Now that we have our set of lengths, we can assume a = 15:
s = [2,4,6]
Output: 5
"""

"""
我们仔细观察题目里ss数组生成的式子，我们可以发现s数组是递增的
很显然，当s_j 越来越小的时候，s_i 的上界就越来越大。因此，我们可以使用双指针的做法来统计答案。
右指针指向j，左指针指向i，随着j的减小，i越来越大。
"""
class Solution:
    """
    @param s0: the number s[0]
    @param n: the number n
    @param k: the number k
    @param b: the number b
    @param m: the number m
    @param a: area
    @return: the way can paint the ceiling
    """
    def painttheCeiling(self, s0, n, k, b, m, a):
        
        walls = [s0]
        temp = s0
        result = 0

        for i in range(1, n):
            temp = (k * temp + b) % m + 1 + temp
            walls.append(temp)

        left, right = 0, len(walls) - 1
        while left < len(walls) and right >= 0:
            if walls[left] * walls[right] <= a:
                result += right + 1
                left += 1
            else:
                right -= 1

        return result