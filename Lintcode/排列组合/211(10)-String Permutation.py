# 211 · String Permutation

"""Given two strings, write a method to decide if one is a permutation of the other."""
class Solution:
    """
    @param A: a string
    @param B: a string
    @return: a boolean
    """
    def Permutation(self, A, B):
        return sorted(A) == sorted(B)



# 10 · String Permutation II
"""Given a string, find all permutations of it without duplicates."""

class Solution:
    """
    @param str: A string
    @return: all permutations
    """
    def stringPermutation2(self, str):
        chars = sorted(list(str))
        visited = [False] * len(chars)
        permutations = []
        self.dfs(chars, visited, "", permutations)
        return permutations
        
    def dfs(self, chars, visited, permutation, permutations):
        if len(permutation) == len(chars):
            permutations.append(permutation)
            return
        
        for i in range(len(chars)):
            if visited[i]:
                continue
            
            if i > 0 and chars[i] == chars[i-1] and not visited[i-1]:
                continue
            
            visited[i] = True
            self.dfs(chars, visited, permutation + chars[i], permutations)
            visited[i] = False