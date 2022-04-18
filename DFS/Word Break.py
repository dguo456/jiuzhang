# 107 · Word Break
# 582 · Word Break II
# 683 · Word Break III

#######################################################################################################


# 107 · Word Break
"""
Given a string s and a dictionary of words dict, determine if s can be broken into a space-separated 
sequence of one or more dictionary words.

Input:
s = "lintcode"
dict = ["lint", "code"]
Output: true
"""
from collections import deque

class Solution:
    """
    @param: s: A string
    @param: dict: A dictionary of words dict
    @return: A boolean
    """
    # Method.1      使用《九章算法班》中讲过的划分型动态规划算法
    #               state: dp[i] 表示前 i 个字符是否能够被划分为若干个单词
    #               function: dp[i] = or{dp[j] and j + 1~i 是一个单词}
    def wordBreak(self, s, wordSet):
        if not s:
            return True
            
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        max_length = max([
            len(word)
            for word in wordSet
        ]) if wordSet else 0
        
        for i in range(1, n + 1):
            for l in range(1, max_length + 1):
                if i < l:
                    break
                if not dp[i - l]:
                    continue
                word = s[i - l:i]
                if word in wordSet:
                    dp[i] = True
                    break
        
        return dp[n]



    # Method.2      记忆化搜索  提前算了下wordDict里的最大单词长度，可以少走一些，会TLE
    def word_break(self, s, word_set) -> bool:
        if not word_set:
            return not s
        max_length = len(max(word_set, key=len))
        return self.dfs(s, word_set, {}, max_length)

    def dfs(self, s, word_dict, memo, max_length):
        if len(s) == 0:
            return True
        if s in memo:
            return memo[s]

        for i in range(1, min(max_length, len(s)) + 1):
            prefix = s[:i]
            if prefix not in word_dict:
                continue
            
            canBreak = self.dfs(s[i:], word_dict, memo, max_length)
            if canBreak:
                memo[s] = True
                return canBreak
        memo[s] = False

        return False


    
    # Method.3      BFS  求是否能抵達，可以使用BFS，加上對find next段落做優化（限制在dict的長度內搜尋)，即可通過
    def wordBreak(self, s, word_set):
        if not s:
            return True

        n = len(s)
        max_length = 0
        for word in word_set:
            max_length = max(len(word), max_length)
        if max_length == 0:
            return False

        queue = deque([0])
        visited = set([0])

        while queue:
            start = queue.popleft()

            for end in range(start+1, min(start+1+max_length, n) + 1):
                if end in visited:
                    continue
                if s[start: end] in word_set:
                    if end == n:
                        return True
                    queue.append(end)
                    visited.add(end)

        return False






# 582 · Word Break II
"""
Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where 
each word is a valid dictionary word. Return all such possible sentences.

Input:  "lintcode", ["de","ding","co","code","lint"]
Output: ["lint code", "lint co de"]
Explanation:
insert a space is "lint code", insert two spaces is "lint co de".
"""
class Solution:
    """
    @param s: A string
    @param dict: A set of word
    @return: the number of possible sentences.
    """
    # Method.1      DFS + memo
    def word_break3(self, s, dict) -> int:
        if not s or not dict:
            return 0

        max_length, lower_case_dict = self.initialize(dict)
        return self.dfs(s.lower(), lower_case_dict, max_length, 0, {})

    def dfs(self, s, d, max_length, index, memo):
        if index == len(s):
            return 1
        if index in memo:
            return memo[index]

        memo[index] = 0
        for i in range(index, len(s)):
            if i + 1 - index > max_length:
                break
            word = s[index: i+1]
            if word not in d:
                continue
            memo[index] += self.dfs(s, d, max_length, i+1, memo)

        return memo[index]

    def initialize(self, d):
        max_length = 0
        lower_case_dict = set()
        for word in d:
            max_length = max(len(word), max_length)
            lower_case_dict.add(word.lower())

        return max_length, lower_case_dict


    
    # Method.2      DP
    def wordBreak3(self, s, dict):
        if not s or not dict:
            return 0

        n, hashset = len(s), set()
        s_lower = s.lower()

        for d in dict:
            hashset.add(d.lower())

        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                sub = s_lower[i: j+1]
                if sub in hashset:
                    dp[i][j] = 1

        for i in range(n):
            for j in range(i, n):
                for k in range(i, j):
                    dp[i][j] += dp[i][k] * dp[k + 1][j]

        return dp[0][-1]



    # 优化
    def wordBreak3(self, s, dict):
        if not s or not dict:
            return 0

        # 将字符全部转化为小写，并将dict转换成hash_set存储，降低判断子串存在性的时间复杂度
        n, hashset = len(s), set()
        s = s.lower()

        for d in dict:
            hashset.add(d.lower())

        # dp[i]表示s[0:i] (不含s[i])的拆分方法数
        dp = [0 for _ in range(n + 1)]

        # dp[0]表示空串的拆分方法数
        dp[0] = 1

        for i in range(n):
            for j in range(i,n):
                # 若存在匹配，则进行状态转移
                if s[i: j+1] in hashset:
                    dp[j+1] += dp[i]

        return dp[n]