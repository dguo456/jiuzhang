# 200 · Longest Palindromic Substring
"""
Given a string S, find the longest palindromic substring in S, and there exists one unique substring.
Input:"abcdzdcab"
Output:"cdzdc"
"""

# 时间复杂度
# 枚举回文中心，复杂度 O(n)。
# 向两边延展并 check，复杂度 O(n)。
# 总时，时间复杂度为 O(n^2)。
# 空间复杂度
# 不需要额外变量，空间复杂度为 O(1)。
class Solution:
    """
    @param s: input string
    @return: a string as the longest palindromic substring
    """
    def longestPalindrome(self, s):
        if not s or len(s) < 2:
            return s

        result = ""
        for center in range(len(s)):
            # 这里因为不确定substring的长度是单数还是双数，所以需要查看两次
            palindrome = self.get_palindrom(s, center, center)
            if len(palindrome) > len(result):
                result = palindrome
            palindrome = self.get_palindrom(s, center, center+1)
            if len(palindrome) > len(result):
                result = palindrome

        return result

    def get_palindrom(self, s, left, right):
        while left >= 0 and right <= len(s) - 1:
            if s[left] != s[right]:
                break
            left -= 1
            right += 1

        return s[left + 1: right]