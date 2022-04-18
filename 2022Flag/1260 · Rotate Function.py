# 1260 · Rotate Function
"""

"""
# 假设原数组的元素和为sum，不难发现：
# F[i] = F[i-1] + sum - n*a[n-i]
class Solution:
    """
    @param a: an array
    @return: the maximum value of F(0), F(1), ..., F(n-1)
    """
    def max_rotate_function(self, a) -> int:
        if not a:
            return 0

        s = sum(a)
        curr = sum(index*value for index, value in enumerate(a))
        max_val = curr

        for i in range(1, len(a)):
            curr += s - len(a)*a[-i]
            max_val = max(curr, max_val)

        return max_val