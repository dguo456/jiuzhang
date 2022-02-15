# 250 · Special Palindrome String
"""
You have a list of paired ambigram letter, Given a string, return true if it is a 
palindrome string which letters in it can be replaced by another corresponding letters.
Input: ambigram=["at", "by", "yh", "hn", "mw", "ww"], word="swims"
Output: true
Explanation: "w" can be replaced by "m" and the string changes to "smims" which is palindrome, so it is true.
"""
from collections import defaultdict

class Solution:
    """
    @param ambigram: A list of paired ambigram letter.
    @param word: A string need to be judged.
    @return: If it is special palindrome string, return true.
    """
    def ispalindrome(self, ambigram, word):

        counter = defaultdict(set)
        for pair in ambigram:
            counter[pair[0]].add(pair[1])
            counter[pair[1]].add(pair[0])
 
        left, right = 0, len(word) - 1
        while left <= right:
            if word[left] != word[right]:
                if not self.is_palindrome(word, counter, left, right):
                    return False 

            left += 1
            right -= 1

        return True 

    def is_palindrome(self, word, counter, left, right):
        if word[left] not in counter or word[right] not in counter:
            return False 

        for l in counter[word[left]]:
            if l in counter[word[right]] or l == word[right]:
                return True 
        for r in counter[word[right]]:
            if r in counter[word[left]] or r == word[left]:
                return True 
        
        return False