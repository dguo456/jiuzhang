# 141 · Sqrt(x)
"""
Implement int sqrt(int x).
Compute and return the square root of x.

Input:  3
Output: 1
Explanation: return the largest integer y that y*y <= x. 
Challenge: O(logn) time complexity
"""
class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """
    def sqrt(self, x):

        start, end = 0, x

        while start + 1 < end:
            root = (start + end) // 2
            if root * root < x:
                start = root
            else:
                end = root
        
        if end * end <= x:
            return end
        return start



# 586 · Sqrt(x) II
"""
Compute and return the square root of x and x >= 0     
The accuracy is kept at 12 decimal places.
"""
class Solution:
    """
    @param x: a double
    @return: the square root of x
    """
    def sqrt(self, x):
        # need to compare x with 1
        if x >= 1:
            start, end = 1, x
        else:
            start, end = x, 1

        # compare with above while condition
        while end - start > 1e-10:
            mid = (start + end) / 2
            if mid * mid < x:
                start = mid
            else:
                end = mid

        return start





# 777 · Valid Perfect Square
"""
Given a positive integer num, write a function which returns True if num is a perfect square else False
Input: num = 16
Output: True
Explanation:  sqrt(16) = 4
"""
class Solution:
    """
    @param num: a positive integer
    @return: if num is a perfect square else False
    """
    def isPerfectSquare(self, num):
        
        # return num ** 0.5 == int(num ** 0.5)

        start, end = 0, num
        while start + 1 < end:
            mid = (start + end) // 2
            if mid * mid <= num:
                start = mid
            else:
                end = mid

        result = start
        if start * start < num:
            result = end
        
        return result * result == num