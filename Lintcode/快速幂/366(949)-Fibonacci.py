# 366 · Fibonacci
"""
Find the Nth number in Fibonacci sequence.

A Fibonacci sequence is defined as follow:

The first two numbers are 0 and 1.
The i th number is the sum of i-1 th number and i-2 th number.
The first ten numbers in Fibonacci sequence is:

0, 1, 1, 2, 3, 5, 8, 13, 21, 34 ...
"""
class Solution:
    def fibonacci(self, n):
        a = 0
        b = 1
        for i in range(n - 1):
            a, b = b, a + b
        return a



"""
算法: 矩阵快速幂

根据题目要求，要求算出斐波那契数列的第n项的末尾4位数字实际上就是Fn%10000,总所周知斐波那契数列的增长速度很快，
那么我们按照Fn=Fn-1+Fn-2这个公式计算Fn需要O(n)的复杂度，显然我们遇到一个很大的n时，空间和时间都不能满足计算，
我们这时候可以考虑用矩阵快速幂来计算Fn

快速幂:

这是一种简单而有效的小算法，它可以以O(logn)的时间复杂度计算乘方
举个例子，我们计算7^10，我们把10写成二进制的形式，也就是 (1010)2
现在问题转变成了计算7^(1010)2，显然我们可以将7^(1010)2拆分成7^(1000)2，7^(10)2。
实际上，对于任意的整数，我们都可以把它拆成若干个7^(1000....)2的形式相乘。而这恰好就是7^1、7^2、7^4……我们只需不断把底数平方就可以算出答案
快速幂的进一步就是矩阵快速幂，两者的区别就是，一个是数字，一个是矩阵
"""
class Matrix:
    def __init__(self): # 零矩阵
        self.mat = [[0 for j in range(2)] for i in range(2)]
    def identityMatrix(self): #单位矩阵
        for i in range(2):
            for j in range(2):
                if i == j:
                    self.mat[i][j] = 1
                else:
                    self.mat[i][j] = 0
    def __mul__(self, m): # 矩阵乘法的计算
        tmp = Matrix()
        for i in range(2):
            for k in range(2):
                if(self.mat[i][k] == 0):
                    continue
                for j in range(2):
                    tmp.mat[i][j] += self.mat[i][k] * m.mat[k][j]
                    tmp.mat[i][j] %= 10000
        return tmp
        
class Solution:
    """
    @param n: an integer
    @return: return a string
    """
    def qpow(self, A, a): # 矩阵快速幂
        B = Matrix()
        B.identityMatrix()
        while a > 0:
            if(a % 2 == 1):
                B = B * A
            A = A * A
            a //= 2
        return B
    def lastFourDigitsOfFn(self, n):
        if(n == 0):
            return 0
        A = Matrix()
        # 构造矩阵
        A.mat = [[1, 1], [1, 0]]
        m = self.qpow(A, n - 1)
        return m.mat[0][0]