
class TrieNode:
    def __init__(self,):
        self.children = {}
        self.is_word = False
        self.word = None
        self.prefix_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def get_root(self):
        return self.root

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.prefix_count += 1
        node.is_word = True
        node.word = word

class Solution:
    """
    @param stringArray: a string array
    @return: return every strings'short peifix
    """
    def ShortPerfix(self, stringArray):
        trie = Trie()
        results = []

        for word in stringArray:
            trie.insert(word)

        for word in stringArray:
            results.append(self.get_unique_prefix(trie.get_root(), word))

        return results

    def get_unique_prefix(self, root, word):
        node = root
        for i in range(len(word)):
            if node.prefix_count == 1:
                return word[:i]
            node = node.children[word[i]]
        return word
