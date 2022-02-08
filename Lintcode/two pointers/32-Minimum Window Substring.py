# 32 · Minimum Window Substring
"""
Given two strings source and target. Return the minimum substring of source which contains each char of target.
You are guaranteed that the answer is unique.
target may contain duplicate char, while the answer need to contain at least the same number of that char.
"""

# Input:
# source = "adobecodebanc"
# target = "abc"
# Output:
# "banc"


"""
Method.1
本题采用滑窗法，滑窗法是双指针技巧，指针left和right分别指向窗口两端，从左向右滑动，实施维护这个窗口。
我们的目标是找到source中涵盖target全部字母的最小窗口，即为最小覆盖子串。
"""
from collections import defaultdict

class Solution:
    """
    @param source : A string
    @param target: A string
    @return: A string denote the minimum window, return "" if there is no such a string
    """
    def minWindow(self, source , target):
        # 初始化counter_s和counter_t
        counter_s = defaultdict(int)
        counter_t = defaultdict(int)
        for ch in target:
            counter_t[ch] += 1
        left = 0
        valid = 0
        # 记录最小覆盖子串的起始索引及长度
        start = -1
        minlen = float('inf')
        for right in range(len(source)):
            # 移动右边界, ch 是将移入窗口的字符
            ch = source[right]
            if ch in counter_t:
                counter_s[ch] += 1
                # 这里判断valid必须是相等，不能是大于等于
                if counter_s[ch] == counter_t[ch]:
                    valid += 1
            
            # 判断左侧窗口是否要收缩
            while valid == len(counter_t):
                # 更新最小覆盖子串
                if right - left < minlen:
                    minlen = right - left
                    start = left
                # left_ch 是将移出窗口的字符
                left_ch = source[left]
                # 左移窗口
                left += 1
                # 进行窗口内数据的一系列更新
                if left_ch in counter_s:
                    counter_s[left_ch] -= 1
                    if counter_s[left_ch] < counter_t[left_ch]:
                        valid -= 1
        # 返回最小覆盖子串,start等于-1意味着right在遍历了整个source之后left都没动，就根本没有运行while循环
        if start == -1:
            return ""
        return source[start: start + minlen + 1]



"""
Method.2        使用同向双指针模板
targetHash 里存了 target 里每个字符和对应的次数次数
hash 里存了当前 i ~ right 之间的字符和对应的出现次数
当 hash 里包含了所有 targetHash 里的字符和其对应出现次数的时候，就是满足条件的字符串
"""
class Solution:
    """
    @param source : A string
    @param target: A string
    @return: A string denote the minimum window, return "" if there is no such a string
    """
    def minWindow(self, source , target):
        if source is None:
            return ""
            
        targetHash = self.getTargetHash(target)
        targetUniqueChars = len(targetHash)
        matchedUniqueChars = 0
        
        hash = {}
        n = len(source)
        right = 0
        minLength = n + 1
        minWindowString = ""
        for left in range(n):
            while right < n and matchedUniqueChars < targetUniqueChars:
                char = source[right]
                if char in targetHash:
                    hash[char] = hash.get(char, 0) + 1
                    if hash[char] == targetHash[char]:
                        matchedUniqueChars += 1
                right += 1
                
            if right - left < minLength and matchedUniqueChars == targetUniqueChars:
                minLength = right - left
                minWindowString = source[left: right]
                
            left_char = source[left]
            if left_char in targetHash:
                if hash[left_char] == targetHash[left_char]:
                    matchedUniqueChars -= 1
                hash[left_char] -= 1
                
        return minWindowString
    
    def getTargetHash(self, target):
        hash = {}
        for c in target:
            hash[c] = hash.get(c, 0) + 1
        return hash