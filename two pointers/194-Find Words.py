# 194 · Find Words
"""
Given a string str and a dictionary dict, you need to find out which words in the dictionary 
are subsequences of the string and return those words. The order of the words returned 
should be the same as the order in the dictionary.

Input:
str="bcogtadsjofisdhklasdj"
dict=["book","code","tag"]
Output:     ["book"]
"""

# 时间复杂度：O(n*m), 空间复杂度：O(n*m)
class Solution:
    """
    @param str: the string
    @param dict: the dictionary
    @return: return words which  are subsequences of the string
    """
    def findWords(self, str, dict):
        if not str or len(str) == 0:
            return []

        results = []
        for word in dict:
            if self.is_subsequence(word, str):
                results.append(word)

        return results

    def is_subsequence(self, word, string):
        if len(string) < len(word):
            return False

        index = 0
        for l in string:
            if l == word[index]:
                index += 1
            if index == len(word):
                return True

        return False

    # 也可以用同向双指针做，但是while的时间效率低于for，对于python来说
    # def is_subsequence(self, word, string):
    #     i, j = 0, 0
    #     while i < len(string) and j < len(word):
    #         if (string[i] == word[j]):
    #             i += 1
    #             j += 1
    #         else:
    #             i += 1
    #     return j == len(word)





# Method.2              用hashmap和二分法进行优化
"""
解法一为什么慢？是因为 str 被遍历了非常多次，每次遍历都要一个一个地往后找匹配的字符。
那怎么优化查找速度呢？可以考虑使用一个哈希表将str中的每种字符的出现位置记录下来，比如字符串 "baccab"，
可以被记录为: {'a': [1, 4], 'b': [0, 5], 'c': [2, 3]}。
那么匹配的时候，就可以通过直接查找对应的字符的出现位置来匹配，例如字符串 "abc":

开始是 'a'，出现的位置是 [1, 4]，使用 str[1] = 'a'，
然后是 'b'，出现的位置是 [0, 5]，由于刚才使用了 str[1] = 'a'，这时就不能使用 str[0] = 'b'，所以使用 str[5] = 'b'，
然后是 'c'，出现的位置是 [2, 3]，由于刚才使用了 str[5] = 'b'，所以字符 'c' 失配，也就是 "abc" 不是 "baccab" 的子序列。
如果使用遍历查找字符的出现位置，那么最坏的复杂度仍然是 O(n*m)，所以此处使用二分法查找出现位置即可。
"""
# 时间复杂度：O(m*log(n)), 空间复杂度：O(n*m)
class Solution:
    """
    @param str: the string
    @param dict: the dictionary
    @return: return words which are subsequences of the string
    """
    def findWords(self, str, dict):
    	# write your code here
        result = []
        positions = {i: [] for i in 'abcdefghijklmnopqrstuvwxyz'}
        for i in range(len(str)):
            positions[str[i]].append(i)
        # print(positions)
        for word in dict:
            if self.is_subsequence(str, word, positions):
                result.append(word)
        
        return result

    def is_subsequence(self, s, t, positions):
        i, j = 0, 0
        while i < len(s) and j < len(t):
            i = self.find_next_position(t[j], i, positions)
            if i == -1:
                break
            j += 1

        return j == len(t)

    def find_next_position(self, char, index, positions):
        if not positions[char]:
            return -1
        left, right = 0, len(positions[char]) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if positions[char][mid] <= index:
                left = mid
            else:
                right = mid

        if index <= positions[char][left]:
            return positions[char][left]
        if index <= positions[char][right]:
            return positions[char][right]
        return -1