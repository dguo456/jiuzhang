# 384 · Longest Substring Without Repeating Characters
"""
Given a string, find the length of the longest substring without repeating characters.
Input: "abcabcbb"
Output: 3
"""

class Solution:
    """
    @param s: a string
    @return: an integer
    """
    def lengthOfLongestSubstring(self, s):
        if not s or len(s) == 0:
            return 0

        unique_chars = set([])
        right = 0
        n = len(s)
        result = 0

        for i in range(n):
            while right < n and s[right] not in unique_chars:
                unique_chars.add(s[right])
                right += 1
                
            result = max(right - i, result)
            # 这里注意可以直接remove掉s[i]，因为进set里的一定是不重复的而且是下一个for循环要用的
            unique_chars.remove(s[i])
            
        return result





# 386 · Longest Substring with At Most K Distinct Characters
"""
Given a string S, find the length of the longest substring T that contains at most k distinct characters.
Input: S = "eceba" and k = 3
Output: 4
Explanation: T = "eceb"
"""
# Method.1
class Solution:
    """
    @param s: A string
    @param k: An integer
    @return: An integer
    """
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if not s or len(s) == 0:
            return 0

        counter = {}
        n = len(s)
        left = 0
        result = 0

        for right in range(n):
            counter[s[right]] = counter.get(s[right], 0) + 1

            # 这里for循环right，等right不再走了，再用left去追right，和sliding window的思想一样
            while left <= right and len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1

            result = max(right - left + 1, result)

        return result



# Method.2
class Solution:
    """
    @param s: A string
    @param k: An integer
    @return: An integer
    """
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if not s or len(s) == 0 or k == 0:
            return 0
            
        counter = {}
        n = len(s)
        right = 0
        result = 0

        for left in range(n):
            
            while right < n and len(counter) <= k:
                counter[s[right]] = counter.get(s[right], 0) + 1
                # 这里注意如果超过了counter的长度，需要把超过的字符去掉，right不再更新
                if len(counter) > k:
                    del counter[s[right]]
                    break

                right += 1

            result = max(right - left, result)

            counter[s[left]] -= 1
            if counter[s[left]] == 0:
                del counter[s[left]]

        return result





# 1375 · Substring With At Least K Distinct Characters
"""
Given a string S with only lowercase characters.
Return the number of substrings that contains at least k distinct characters.

Input: S = "abcabcabcabc", k = 3
Output: 55
Explanation: Any substring whose length is not smaller than 3 contains a, b, c.
    For example, there are 10 substrings whose length are 3, "abc", "bca", "cab" ... "abc"
    There are 9 substrings whose length are 4, "abca", "bcab", "cabc" ... "cabc"
    ...
    There is 1 substring whose length is 12, "abcabcabcabc"
    So the answer is 1 + 2 + ... + 10 = 55.
"""
class Solution:
    """
    @param s: a string
    @param k: an integer
    @return: the number of substrings there are that contain at least k distinct characters
    """
    def k_distinct_characters(self, s: str, k: int) -> int:
        
        counter = {}
        n = len(s)
        left = 0
        result = 0

        for right in range(n):
            counter[s[right]] = counter.get(s[right], 0) + 1

            while left <= right and len(counter) >= k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1
                    
            # "abcdefagebsge", 如果”cdef“符合，那么”abcdef“和”bcdef“肯定也都符合
            if len(counter) == k - 1 and left > 0 and s[left - 1] not in counter:
                result += left
                
        return result
        





# 928 · Longest Substring with At Most Two Distinct Characters
"""
Given a string, find the length of the longest substring T that contains at most 2 distinct characters.
Input: “eceba”
Output: 3
Explanation:
T is "ece" which its length is 3.
"""
class Solution:
    """
    @param s: a string
    @return: the length of the longest substring T that contains at most 2 distinct characters
    """
    def lengthOfLongestSubstringTwoDistinct(self, s):
        if not s or len(s) == 0:
            return 0

        counter = {}
        n = len(s)
        left = 0
        result = 0

        for right in range(n):
            counter[s[right]] = counter.get(s[right], 0) + 1

            while left <= right and len(counter) > 2:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    del counter[s[left]]
                left += 1

            result = max(right - left + 1, result)

        return result