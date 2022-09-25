# 139 · Word Break
# 140 · Word Break II
#######################################################################################################


# 139 · Word Break
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






# 140 · Word Break II
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
    # Approach 1: Top-Down DP
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        @lru_cache(None)
        def backtrack(index: int) -> List[List[str]]:
            if index == len(s):
                return [[]]
            ans = list()
            for i in range(index + 1, len(s) + 1):
                word = s[index:i]
                if word in wordSet:
                    nextWordBreaks = backtrack(i)
                    for nextWordBreak in nextWordBreaks:
                        ans.append(nextWordBreak.copy() + [word])
            return ans
        
        wordSet = set(wordDict)
        breakList = backtrack(0)
        return [" ".join(words[::-1]) for words in breakList]
    
    # Approach 1: Top-Down DP (memo)
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        if not s or not wordDict:
            return []
        
        word_set = set(wordDict)
        max_len = max(len(word) for word in wordDict)
        self.memo = defaultdict(list)
        
        self.dfs(s, word_set, max_len)
        return [" ".join(word) for word in self.memo[s]]
    
    def dfs(self, s, word_set, max_len):
        if not s:
            return [[]]
        if s in self.memo:
            return self.memo[s]
        
        for end in range(1, min(max_len, len(s)) + 1):
            prefix = s[: end]
            if prefix not in word_set:
                continue
            for sub_s in self.dfs(s[end:], word_set, max_len):
                self.memo[s].append([prefix] + sub_s)
                
        return self.memo[s]
    
    
    # Approach 2: Bottom-Up DP
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # quick check on the characters, 
        # otherwise it would exceed the time limit for certain test cases.
        if set(Counter(s).keys()) > set(Counter("".join(wordDict)).keys()):
            return []

        wordSet = set(wordDict)
        dp = [[]] * (len(s) + 1)
        dp[0] = [""]

        for endIndex in range(1, len(s)+1):
            sublist = []
            for startIndex in range(0, endIndex):
                word = s[startIndex: endIndex]
                if word in wordSet:
                    for subsentence in dp[startIndex]:
                        sublist.append((subsentence + ' ' + word).strip())

            dp[endIndex] = sublist

        return dp[len(s)]
