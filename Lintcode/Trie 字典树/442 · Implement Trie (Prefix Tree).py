class TrieNode:
    def __init__(self,):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        @param: word: a word
        @return: nothing
        """
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        
        node.is_word = True

    def search(self, word):
        """
        @param: word: A string
        @return: if the word is in the trie.
        """
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
            # if ch not in node.children:
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
            
        return node is not None