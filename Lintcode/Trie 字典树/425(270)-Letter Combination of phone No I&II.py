# 425 · Letter Combinations of a Phone Number
KEYBOARDS = {
    
    '2': ['a', 'b', 'c'],
    '3': ['d', 'e', 'f'],
    '4': ['g', 'h', 'i'],
    '5': ['j', 'k', 'l'],
    '6': ['m', 'n', 'o'],
    '7': ['p', 'q', 'r', 's'],
    '8': ['t', 'u', 'v'],
    '9': ['w', 'x', 'y', 'z']
}

class Solution:
    """
    @param digits: A digital string
    @return: all posible letter combinations
    """
    def letterCombinations(self, digits):
        if not digits:
            return []
        self.results = []
        self.dfs(digits, '', 0)
        return self.results
        
    def dfs(self, digits, substring, index):
        if index == len(digits):
            self.results.append(substring)
            return
        
        for letter in KEYBOARDS[digits[index]]:
            self.dfs(digits, substring+letter, index+1)



# 270 · Letter Combinations of a Phone Number II
"""
这个题不需要 DFS，只需要建 Trie 就行。
存在 Trie 里的不是单词，而是单词转成数字之后的字符串。
比如词典里有 ["a", "abc", "de", "fg"]，存在 Trie 里的就是 ["2", "222", "33", "34"]
之后的话 for queries 里的每个 query，在 Trie 里沿着 query 走到对应的节点上看一下计数就行。

Example:
Input: query = ["2", "3", "4"]
        dict = ["a","abc","de","fg"]
Output:[2,2,0]
Explanation:  "a" "abc" match "2"     "de" "fg" match "3"      no word match "4"
"""

REVERSE_KEYBOARD = {
    "a": "2", "b": "2", "c": "2",
    "d": "3", "e": "3", "f": "3",
    "g": "4", "h": "4", "i": "4",
    "j": "5", "k": "5", "l": "5",
    "m": "6", "n": "6", "o": "6",
    "p": "7", "q": "7", "r": "7", "s": "7",
    "t": "8", "u": "8", "v": "8",
    "w": "9", "x": "9", "y": "9", "z": "9",
}

class TrieNode:
    def __init__(self):
        self.children = {}
        self.prefix_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.prefix_count += 1
    
    
class Solution:
    """
    @param queries: the queries
    @param dict: the words
    @return: return the queries' answer
    """
    def letterCombinationsII(self, queries, dict):
        trie = Trie()
        for word in dict:
            digit_word = ''.join([
                REVERSE_KEYBOARD[c]
                for c in word
            ])
            trie.insert(digit_word)
            
        results = []
        for query in queries:
            node = trie.root
            for char in query:
                if char not in node.children:
                    node = None
                    break
                node = node.children[char]
            results.append(node.prefix_count if node else 0)
            
        return results