# 428 · Pow(x, n) 求幂
"""Implement pow(x, n). (n is an integer.)"""

class Solution:
    """
    @param x {float}: the base number
    @param n {int}: the power number
    @return {float}: the result
    """
    def myPow(self, x, n):
        if n < 0 :
            x = 1 / x  
            n = -n

        result = 1
        tmp = x

        while n != 0:
            if n % 2 == 1:
                result *= tmp
            tmp *= tmp
            n //= 2

        return result



# 140 · Fast Power  快速幂
"""
Calculate the a^n % b where a, b and n are all 32bit non-negative integers.
Input:
    a = 3  b = 7  n = 5
Output: 5
    3 ^ 5 % 7 = 5
"""
class Solution:
    """
    @param a: A 32bit integer
    @param b: A 32bit integer
    @param n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        result = 1
        tmp = a

        while n != 0:
            if n % 2 == 1:
                result = (result * tmp) % b
            tmp = (tmp * tmp) % b
            n = n // 2

        return result % b 



# 1275 · Super Pow
"""
Your task is to calculate a^b mod 1337 where a is a positive integer and 
b is an extremely large positive integer given in the form of an array.
The length of b is in range [1, 1100]

Example1:
Input:
    a = 2
    b = [3]
Output:
    8

Example2:
Input:
    a = 2
    b = [1,0]
Ouput:
    1024

具体公式: https://www.lintcode.com/problem/1275/solution/21324
"""
class Solution:
    """
    @param a: the given number a
    @param b: the given array
    @return: the result
    """
    def superPow(self, a, b):
        if a == 0:
            return 0

        result = 1
        func = lambda x: x % 1337

        for num in b:
            result = func(func(result ** 10) * func(a ** num))

        return result
