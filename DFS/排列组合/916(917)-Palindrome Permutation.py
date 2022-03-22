# 916 · Palindrome Permutation
"""
Example:

Input: s = "aab"
Output: True
Explanation: 
"aab" --> "aba"
"""
from collections import Counter

class Solution:
    """
    @param s: the given string
    @return: if a permutation of the string could form a palindrome
    """
    def canPermutePalindrome(self, s):
        
        return sum(v % 2 for v in Counter(s).values()) < 2



# 917 · Palindrome Permutation II
"""
Given a string s, return all the palindromic permutations (without duplicates) of it. 
Return an empty list if no palindromic permutation could be form.

Example1
Input: s = "aabb"
Output: ["abba","baab"]

Example2
Input: "abc"
Output: []
"""
class Solution:
    """
    @param s: the given string
    @return: all the palindromic permutations (without duplicates) of it
    """
    def generatePalindromes(self, s):

        counter = {}
        # odds = filter(lambda x: x % 2, counter.values())
        for c in s:
            counter[c] = counter.get(c, 0) + 1
        odds =  [c for c in counter if counter[c] % 2 == 1]
        if len(odds) > 1:
            return []

        half_s = []
        for c in counter:
            half_s.extend([c] * (counter[c] // 2))  

        visited = [False] * len(half_s)
        permutations = []
        self.dfs(half_s, visited, "", permutations)

        results = []
        # 这里优化是因为正常的代码会超时，只用palindrome的一半传入做dfs
        for permutation in permutations:
            if odds:
                results.append(permutation + odds[0] + permutation[::-1])
            else:
                results.append(permutation + permutation[::-1])

        return results

    def dfs(self, chars, visited, permutation, permutations):
        if len(permutation) == len(chars):
            permutations.append(permutation)
            return

        for i in range(len(chars)):
            if visited[i]:
                continue

            if i > 0 and chars[i-1] == chars[i] and not visited[i-1]:
                continue

            visited[i] = True
            self.dfs(chars, visited, permutation + chars[i], permutations)
            visited[i] = False