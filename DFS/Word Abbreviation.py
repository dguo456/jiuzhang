# 637 · Valid Word Abbreviation
# 639 · Word Abbreviation
# 648 · Unique Word Abbreviation
# 779 · Generalized Abbreviation
# 890 · Minimum Unique Word Abbreviation

###########################################################################################################


# 637 · Valid Word Abbreviation
"""

"""
class Solution:
    """
    @param word: a non-empty string
    @param abbr: an abbreviation
    @return: true if string matches with the given abbr or false
    """
    # 用两个指针i,j分别从头开始匹配，j遇到数字，就先把数字x解析出来，
    # 然后i移动x位，继续匹配。如果不能匹配就返回false。
    def validWordAbbreviation(self, word, abbr):
        i, j = 0, 0
        while i < len(word) and j < len(abbr):
            if word[i] == abbr[j]:
                i += 1
                j += 1
            elif abbr[j].isdigit() and abbr[j] != '0':
                start = j
                while j < len(abbr) and abbr[j].isdigit():
                    j += 1
                i += int(abbr[start : j])
            else:
                return False

        return i == len(word) and j == len(abbr)





# 639 · Word Abbreviation
"""
Given an array of n distinct non-empty strings, you need to generate minimal possible abbreviations 
for every word following rules below.
1. Begin with the first character and then the number of characters abbreviated, which followed by the 
    last character.
2. If there are any conflict, that is more than one words share the same abbreviation, a longer prefix 
    is used instead of only the first character until making the map from word to abbreviation 
    become unique. In other words, a final abbreviation cannot map to more than one original words.
3. If the abbreviation doesn't make the word shorter, then keep it as original.
4. The return answers should be in the same order as the original array.

Input:
["like","god","internal","me","internet","interval","intension","face","intrusion"]
Output:
["l2e","god","internal","me","i6t","interval","inte4n","f2e","intr4n"]
"""
from collections import defaultdict

class Solution:
    """
    @param dict: an array of n distinct non-empty strings
    @return: an array of minimal possible abbreviations for every word
    """
    # Method.1      做一个和dict对应的abbreviate的数组。也就是说dict[i] => abbreviation[i] 对应。 
    #               然后用一个哈希纪录一下abbreviation出现的次数。遇到多1次的缩写，那就更新这个缩写，
    #               并更新哈希表。更新的方式 通过一个不停增加保留前面字母个数的函数实现
    def wordsAbbreviation(self, dict):
        if not dict:
            return []
            
        n = len(dict)
        prefix = [1] * n
        abbr_count = {}
        results = []
        
        # 先默认没有特殊情况处理，把每一个单词按顺序append进results里，为了保持顺序
        for i in range(n):
            word = dict[i]
            abbr = self.get_abbr(word, prefix[i])
            results.append(abbr)
            abbr_count[abbr] = abbr_count.get(abbr, 0) + 1 

        # 接下来再判断是否有特殊重复单词需要特殊处理
        while True:
            unique = True
            for i in range(n):
                if abbr_count[results[i]] > 1:
                    prefix[i] += 1 
                    new_abbr = self.get_abbr(dict[i], prefix[i])
                    results[i] = new_abbr
                    abbr_count[new_abbr] = abbr_count.get(new_abbr,0) + 1 
                    unique = False 
            if unique:
                break
            
        return results 
        
    def get_abbr(self, word, prefix_len):
        if prefix_len >= len(word) - 2:
            return word
        
        left, right = word[:prefix_len], word[-1]    
        mid_len = len(word) - prefix_len -1 
        return left + str(mid_len) + right


    # Method.2      九章官方答案，精炼版
    def wordsAbbreviation(self, dict):
        self.dict = {}
        self.solve(dict, 0)
        return list(map(self.dict.get, dict))

    def abbr(self, word, size):
        if len(word) - size <= 3:
            return word
        return word[:size + 1] + str(len(word) - size - 2) + word[-1]

    def solve(self, dict, size):
        dlist = defaultdict(list)
        for word in dict:
            dlist[self.abbr(word, size)].append(word)
        for abbr, wlist in dlist.items():
            if len(wlist) == 1:
                self.dict[wlist[0]] = abbr
            else:
                self.solve(wlist, size + 1)






# 648 · Unique Word Abbreviation
"""
An abbreviation of a word follows the form <first letter><number><last letter>. 
Below are some examples of word abbreviations:
Assume you have a dictionary and given a word, find whether its abbreviation is unique 
in the dictionary. A word's abbreviation is unique if no other word from the dictionary 
has the same abbreviation.

Input:
[ "deer", "door", "cake", "card" ]
isUnique("dear")
isUnique("cart")
Output:
false
true
Explanation:
Dictionary's abbreviation is ["d2r", "d2r", "c2e", "c2d"].
"dear" 's abbreviation is "d2r" , in dictionary.
"cart" 's abbreviation is "c2t" , not in dictionary.
"""
class ValidWordAbbr:
    """
    @param: dictionary: a list of words
    """
    def __init__(self, dictionary):
        self.abbr_dict = {}
        for word in dictionary:
            abbr = self.word2Vec(word)
            if abbr not in self.abbr_dict:
                self.abbr_dict[abbr] = set()
            # else:
            self.abbr_dict[abbr].add(word)

    def word2Vec(self, word):
        if len(word) < 3:
            return word
        return word[0] + str(len(word[1:-1])) + word[-1]

    """
    @param: word: a string
    @return: true if its abbreviation is unique or false
    """
    def isUnique(self, word):
        abbr = self.word2Vec(word)

        if abbr not in self.abbr_dict:
            return True

        for word_in_dict in self.abbr_dict[abbr]:
            if word_in_dict != word:
                return False

        return True

# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param = obj.isUnique(word)







# 779 · Generalized Abbreviation
"""
Write a function to generate the generalized abbreviations of a word.(order does not matter)

Input: 
word = "word", 
Output: 
["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", 
    "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
"""
class Solution:
    """
    @param word: the given word
    @return: the generalized abbreviations of a word
    """
    # Method.1      标准 DFS，函数的输入参数里想少写点 boolean flag， 所以稍微慢一点，
    #               在 path 里保持原有 int 属性便于加减
    def generate_abbreviations(self, word: str):
        if not word:
            return []
            
        results = []
        self.dfs(word, [], 0, results)
        return results
    
    def dfs(self, word, path, pos, results):
        if pos == len(word):
            results.append("".join([str(i) for i in path]))
            return
        
        path.append(word[pos])
        self.dfs(word, path, pos + 1, results)
        path.pop()
        
        if path:
            if isinstance(path[-1], int):
                path[-1] += 1
                self.dfs(word, path, pos + 1, results)
                path[-1] -= 1
            else:
                path.append(1)
                self.dfs(word, path, pos + 1, results)
                path.pop()
        else:
            path.append(1)
            self.dfs(word, path, pos + 1, results)
            path.pop()

        return


    # Method.2      abbrCount用来记录到目前为止已经abbrevite的数量，
    #               每次进入下一次dfs之前把curWord处理好，相对要容易理解一些
    def generateAbbreviations(self, word):
        if not word:
            return []
        
        result = []
        self.dfs("", 0, 0, result, word)
        return result 
    
    def dfs(self, curWord, index, abbrCount, result, word):
        if index == len(word):
            result.append(curWord)
            return 
        
        # do not abbreviate at current index
        self.dfs(curWord + word[index], index + 1, 0, result, word)

        # abbreviate at current index
        if abbrCount > 0:
            # remove the previous count in the string and append the new count into the string 
            curWord = curWord[:-len(str(abbrCount))] + str(abbrCount + 1)
        else:
            # the first abbrevation, hence the abbr string is '1'
            curWord += '1'

        self.dfs(curWord, index + 1, abbrCount + 1, result, word)






# 890 · Minimum Unique Word Abbreviation
"""
A string such as "word" contains the following abbreviations:
["word","1ord","w1rd","wo1d","wor1","2rd","w2d","wo2","1o1d","1or1","w1r1","1o2","2r1","3d","w3","4"]
Given a target string and a set of strings in a dictionary, find an abbreviation of this target string 
with the smallest possible length such that it does not conflict with abbreviations of the strings 
in the dictionary. Each number or letter in the abbreviation is considered length = 1. 
For example, the abbreviation "a32bc" has length = 5.

Example 1:
Input: "apple",["blade"]
Output: "a4"
Explanation: Because "5" or "4e" conflicts with "blade".

Example 2:
Input: "apple",["plain","amber","blade"]
Output: "1p3"
Explanation: Other valid answers include "ap3", "a3e", "2p2", "3le", "3l1"
"""
class Solution:
    """
    @param target: a target string 
    @param dictionary: a set of strings in a dictionary
    @return: an abbreviation of the target string with the smallest possible length
    """
    # Method.1      暴力做法：把所有单词的缩写都生成到一个 hash set 里。
    # 然后再看看 target 的所有的缩写里最短的没有出现在这个 set 里的缩写是什么。
    # 时间复杂度 O(m * n * 2^m)
    def minAbbreviation(self, target, dictionary):
        global_abbr_set = set()
        for word in dictionary:
            self.add_to_abbr_set(word, global_abbr_set)
        
        target_abbr_set = set()
        self.add_to_abbr_set(target, target_abbr_set)

        shortest_abbr = target
        for abbr in target_abbr_set:
            if abbr in global_abbr_set:
                continue
            if len(abbr) < len(shortest_abbr):
                shortest_abbr = abbr
        return shortest_abbr

    def add_to_abbr_set(self, word, abbr_set):
        for i in range(1, len(word) + 1):
            self.dfs(word, i, word[:i], abbr_set)
            self.dfs(word, i, str(i), abbr_set)

    def dfs(self, word, index, abbr, abbr_set):
        if index >= len(word):
            abbr_set.add(abbr)
            return

        for i in range(index + 1, len(word) + 1):
            if abbr[-1].isdigit():
                curt = word[index: i]
            else:
                curt = str(i - index)
            self.dfs(word, i, abbr + curt, abbr_set)



    # Method.2  算法原理:
    # 稍微优化一点的算法：假如说 target 的缩写是 a1b2c3，可以计算出 a,b,c 在 target 的下标分别是 (0,2,5) 
    # 那么我们求出以下三个集合的合并：
    # 下标0这个位置不是a的所有单词 / 下标2这个位置不是b的所有单词 / 下标5这个位置不是c的所有单词
    # 试想如果存在一个 dictionary 里的单词，这三个位置分别都是 a,b,c 的话（也就是缩写也可以写成 a1b2c3）
    # 那么它一定不在这个合并的集合里。反之，如果这个合并的集合包含了 dictionary 里所有的单词的话，那么意味着
    # 不存在任何一个单词，可以被写成 a1b2c3 的缩写。
    # 算法步骤: 因此从整体的算法如下：
    # 首先预处理出每个位置上和 target 不同的单词集合是哪些，存在一个 index_to_wordset 的哈希表里。
    # 比如 apple 和 amber 在下标1,2,3,4 上都不同，因此 amber 会出现在 index_to_wordset 的 1,2,3,4 
    # 这4个不同的key所找到的 wordset 里。然后找到target 的所有缩写形式，对于每种缩写形成，计算出每个字符的位置上
    # 不同的单词集合的并集，看看这个集合是否是所有的单词。如果是，那就是一个合理的缩写，找到最短的合理的缩写即可。
    # 时间复杂度:   预处理 O(m*n)   主过程 O(2^m * m * n)
    # 从最坏时间复杂度的角度其实没有太多优化，但是实际效率会好一些。因为会减少一些无谓的操作相比于最暴力的算法。
    def minAbbreviation(self, target, dictionary):
        self.index_to_wordset = self.get_index_to_wordset(target, dictionary)
        self.dictionary = dictionary
        self.shortest_abbr = target
        self.dfs_entry(target)
        return self.shortest_abbr

    def get_index_to_wordset(self, target, dictionary):
        index_to_wordset = {}
        for i in range(len(target)):
            word_set = set()
            for word in dictionary:
                if word[i] != target[i]:
                    word_set.add(word)
            index_to_wordset[i] = word_set
        return index_to_wordset

    def check_abbr(self, abbr_parts):
        index = 0
        union_word_set = set()
        for abbr_part in abbr_parts:
            if abbr_part.isdigit():
                index += int(abbr_part)
                continue
            for word in self.index_to_wordset[index]:
                union_word_set.add(word)
            index += 1

        if len(union_word_set) != len(self.dictionary):
            return
        # find a valid abbr
        abbr = ''.join(abbr_parts)
        if len(self.shortest_abbr) > len(abbr):
            self.shortest_abbr = abbr

    def dfs_entry(self, word):
        for i in range(1, len(word) + 1):
            self.dfs(word, i, [str(i)])
            self.dfs(word, i, [word[:i]])

    def dfs(self, word, index, abbr_parts):
        if index >= len(word):
            self.check_abbr(abbr_parts)
            return

        for i in range(index + 1, len(word) + 1):
            if abbr_parts[-1].isdigit():
                curt = word[index: i]
            else:
                curt = str(i - index)
            abbr_parts.append(curt)
            self.dfs(word, i, abbr_parts)
            abbr_parts.pop()