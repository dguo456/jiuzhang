# 1739 · Numbers At Most N Given Digit Set
"""
We have a sorted set of digits D, a non-empty subset of {'1','2','3','4','5','6','7','8','9'}. 
(Note that '0' is not included.)
Now, we write numbers using these digits, using each digit as many times as we want. For example, 
if D = {'1','3','5'}, we may write numbers such as '13', '551', '1351315'.
Return the number of positive integers that can be written (using the digits of D) that are 
less than or equal to N.

Input: D = ["1","3","5","7"], N = 100
Output: 20
Explanation: 
The 20 numbers that can be written are:
1, 3, 5, 7, 11, 13, 15, 17, 31, 33, 35, 37, 51, 53, 55, 57, 71, 73, 75, 77.
"""
class Solution:
    """
    @param D: The sorted set of digits.
    @param N: 
    @return: The number of positive integers that can be written.
    """
    # Method.1      小于N位数的数字直接计算，等于N位数的数组从前向后依次计算
    def atMostNGivenDigitSet(self, D, N):
        N = str(N)
        n = len(N)
        res = sum(len(D) ** i for i in range(1, n))
        
        i = 0
        while i < len(N):
            res += sum(c < N[i] for c in D) * (len(D) ** (n - i - 1))
            if N[i] not in D: 
                break
            i += 1

        return res + (i == n)