# 328 · String Partition                same with 1045 Partition Labels
"""
Given a string with all characters in uppercase, please divide the string into as many parts as possible 
so that each letter appears in only one part. Return an array containing the length of each part.

Input:      "MPMPCPMCMDEFEGDEHINHKLIN"
Output:     [9,7,8]

Explanation:
"MPMPCPMCM"
"DEFEGDE"
"HINHKLIN"
"""
class Solution:
    """
    @param s: a string
    @return:  an array containing the length of each part
    """
    def splitString(self, s):
        if not s and len(s) == 0:
            return s

        d = {}
        for i in range(len(s)):
            # key, value对分别存的是当前的char和这个char在s里出现最右的index
            d[s[i]] = i

        left, right = 0, 0
        results = []

        while right < len(s):
            # 每次外循环设置起始位置为当前char在d中的value作为内层循环的index
            index = d[s[left]]
            while right <= index:       # 这里注意index随时在变化，所以是一个动态的过程
                if d[s[right]] > index:
                    index = d[s[right]]
                right += 1

            results.append(index - left + 1)
            left = right

        return results