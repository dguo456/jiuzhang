# 473 · Add and Search Word - Data structure design
"""
Design a data structure that supports the following two operations: addWord(word) and search(word)
Addword (word) adds a word to the data structure. search(word) can search a literal word or a 
regular expression string containing only letters a-z or .
A "." means it can represent any one letter.

Input:
  addWord("a")
  search(".")
Output:
  true
"""

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
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    """
    @param: word: A word could contain the dot character '.' to represent any one letter.
    @return: if the word is in the data structure.
    """
    def search(self, word):
        return self.helper(self.root, word)
        
    def helper(self, node, word):
        if not word:
            return node.is_word

        char = word[:1]
        if char != '.':
            if char in node.children:
                next_node = node.children[char]
                return self.helper(next_node, word[1:])
            else:
                return False
        else:
            for _, next_node in node.children.items():
                if self.helper(next_node, word[1:]):
                    return True
            return False