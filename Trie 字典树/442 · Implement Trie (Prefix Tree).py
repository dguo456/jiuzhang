class TrieNode:
    def __init__(self,):
        self.children = {}
        self.is_word = False
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        @param: word: a word
        @return: nothing
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_word = True
        node.word = word

    def search(self, word):
        """
        @param: word: A string
        @return: if the word is in the trie.
        """
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return False
            node = node.children[ch]

        return node is not None and node.is_word == True

    def startsWith(self, prefix):
        """
        @param: prefix: A string
        @return: if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return False
            
        return node






# 473 · Add and Search Word - Data structure design
"""
Design a data structure that supports the following two operations: addWord(word) and search(word)
Addword (word) adds a word to the data structure.search(word) can search a literal word or a 
regular expression string containing only letters a-z or .
A . means it can represent any one letter.

Input:
  addWord("a")
  search(".")
Output:
  true
"""

# 可以说, 这道题是 442. 实现 Trie (前缀树) 的升级版.
# 只需要在查询的时候将 '.' 处理为回溯即可, 即遇到 '.' 则需要访问每一个子节点判断是否有匹配.
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class WordDictionary:
    
    def __init__(self):
        self.root = TrieNode()
        
    """
    @param: word: Adds a word into the data structure.
    @return: nothing
    """
    def addWord(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word =True

    """
    @param: word: A word could contain the dot character '.' to represent any one letter.
    @return: if the word is in the data structure.
    """
    def search(self, word):
        if word is None:
            return False
        return self.search_helper(self.root, word, 0)
        
    def search_helper(self, node, word, index):
        if node is None:
            return False
            
        if index >= len(word):
            return node.is_word
        
        char = word[index]
        if char != '.':
            return self.search_helper(node.children.get(char), word, index + 1)
            
        for child in node.children:
            if self.search_helper(node.children[child], word, index + 1):
                return True
                
        return False



    # Method.2      答案中用的全都是把 index 当成参数传入。如果用slicing可以代码更简单一点。
    #               判断用slicing还是用index的条件：除了string外是否还有别的需要用到index
    def search(self, word):
        return self.helper(self.root, word)
        
    def helper(self, node, s):
        if not s:
            return node.is_word

        c = s[:1]
        if c != '.':
            if c in node.children:
                n = node.children[c]
                return self.helper(n, s[1:])
            else:
                return False
        else:
            for _, n in node.children.items():
                if self.helper(n, s[1:]):
                    return True 
            return False