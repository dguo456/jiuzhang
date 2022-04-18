# 123 · Word Search
# 132 · Word Search II
# 1848 · Word Search III
# 635 · Boggle Game

#########################################################################################################


# 123 · Word Search
"""
Given a 2D board and a string word, find if the string word exists in the grid.
The string word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells 
are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

board = ["ABCE","SFCS","ADEE"]
word = "ABCCED"

Output:     true
Explanation:
[
A B C E
S F C S
A D E E
]
(0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,1)
"""

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Solution:
    """
    @param board: A list of lists of character
    @param word: A string
    @return: A boolean
    """
    def exist(self, board, word) -> bool:
        if not board or not word:
            return False

        n, m = len(board), len(board[0])
        visited = [[False for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    visited[i][j] = True    # 此处注意
                    can_be_constructed = self.dfs(board, word, 1, visited, i, j)
                    visited[i][j] = False
                    if can_be_constructed:
                        return True
        return False

    def dfs(self, board, word, matched, visited, x, y):
        if matched == len(word):
            return True

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy

            if not (0 <= next_x < len(board) and 0 <= next_y < len(board[0])):
                continue
            if visited[next_x][next_y] == True:
                continue
            if board[next_x][next_y] != word[matched]:    # 这里matched不会 out of range
                continue

            visited[next_x][next_y] = True
            result = self.dfs(board, word, matched+1, visited, next_x, next_y)
            visited[next_x][next_y] = False

            if result:
                return True

        return False





# 132 · Word Search II
"""
Given a matrix of lower alphabets and a dictionary. Find all words in the dictionary that can be found 
in the matrix. A word can start from any position in the matrix and go left/right/up/down to the 
adjacent position. One character only be used once in one word. No same word in dictionary

Input: ["doaf","agai","dcan"], ["dog","dad","dgdg","can","again"]
Output: ["again","can","dad","dog"]
Explanation:
  d o a f
  a g a i
  d c a n
search in Matrix, so return ["again","can","dad","dog"].
"""
class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: A list of string
    """
    # Method.1      同 word search I
    def word_search_i_i(self, board, words):
        if not board or not words or not words[0]:
            return []

        n, m = len(board), len(board[0])
        visited = [[False for _ in range(m)] for _ in range(n)]
        # results = []          用list也可以，但是据说用set更快
        words_set = set(words)
        results = set()

        for word in words_set:
            for i in range(n):
                for j in range(m):
                    if board[i][j] == word[0]:
                        visited[i][j] = True
                        can_be_constructed = self.dfs(board, word, 1, visited, i, j)
                        visited[i][j] = False
                        # if can_be_constructed and word not in results:
                        #     results.append(word)
                        if can_be_constructed:
                            results.add(word)

        return list(results)

    def dfs(self, board, word, matched, visited, x, y):
        if matched == len(word):
            return True

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy

            if not (0 <= next_x < len(board) and 0 <= next_y < len(board[0])):
                continue
            if visited[next_x][next_y] == True:
                continue
            if board[next_x][next_y] != word[matched]:
                continue

            visited[next_x][next_y] = True
            result = self.dfs(board, word, matched+1, visited, next_x, next_y)
            visited[next_x][next_y] = False

            if result:
                return True

        return False



# Method.2      使用Trie字典树
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None
        
        
class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_word = True
        node.word = word
        
    def search(self, word):	
        node = self.root
        for char in word:
            node = node.children.get(char)
            if node is None:
                return None
                
        return node
        

class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: A list of string
    """
    def wordSearchII(self, board, words):
        if board is None or len(board) == 0:
            return []
            
        trie = Trie()
        for word in words:
            trie.insert(word)

        visited, results = set(), set()

        for i in range(len(board)):
            for j in range(len(board[0])):
                visited.add((i, j))
                c = board[i][j]
                self.search(board, i, j, trie.root.children.get(c), visited, results)
                visited.remove((i, j))
                
        return list(results)
        
    # 字典树上使用 DFS 进行查找
    def search(self, board, x, y, node, visited, result):
        if node is None:
            return
        
        if node.is_word:
            result.add(node.word)
        
        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy
            
            if not (0 <= next_x < len(board) and 0 <= next_y < len(board[0])):
                continue
            if (next_x, next_y) in visited:
                continue
            
            visited.add((next_x, next_y))
            self.search(board, next_x, next_y, node.children.get(board[next_x][next_y]), visited, result)
            visited.remove((next_x, next_y))





# 1848 · Word Search III
"""
Given a matrix of lower alphabets and a dictionary. Find maximum number of words in the dictionary that 
can be found in the matrix in the meantime. A word can start from any position in the matrix and go 
left/right/up/down to the adjacent position. One character only be used once in the matrix. 
No same word in dictionary

Input:  ["doaf","agai","dcan"],  ["dog","dad","dgdg","can","again"]
Output: 2
Explanation:
  d o a f
  a g a i
  d c a n
search in Matrix, you can find `dog` and `can` in the meantime.
"""
# Method.1
# 1. dfs获得字典中存在的单词
# 2. 当获得的字符串为字典中的一个单词时，需要考虑2种结果，
#   a) 将当前字符串作为一个单词，count+1，从当前字符串的起点处的下一个位置开始dfs，查找新单词
#   b) 将当前字符串作为单词的一部分，继续dfs往字符串后面追加字符
# 常规dfs会超时，考虑使用前缀树用来加速，可以使用Trie或者prefix_map
# 该题与635非常相似，唯一区别在于本题要求单词也不能重复，所以需要额外记忆该单词是否已经用过。
class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: return the maximum nunber
    """
    def word_search_i_i_i(self, board, words) -> int:
        if not board or not board[0]:
            return 0

        # 纯dfs解法会超时，加一个prefix给dfs加速
        prefix = self.get_prefix(words)
        n, m = len(board), len(board[0])

        answer = 0
        word_set = set(words)
        for i in range(n):
            for j in range(m):
                # 以i,j为起点，找到单词的数量
                res = [0]
                self.dfs(board, word_set, i, j, i, j, set([(i, j)]), board[i][j], 0, res, prefix)
                answer = max(answer, res[0])

        return answer

    def dfs(self, board, word_set, x, y, start_x, start_y, visited, word, count, res, prefix):
        if word in word_set:
            word_set.remove(word)
            count += 1
            res[0] = max(res[0], count)

            # 从上一轮开始的后一位开始继续找
            for i in range(start_x, len(board)):
                new_start_y = 0
                # 当前行从start_y + 1 开始,其它行从0开始
                if i == start_x:
                    new_start_y = start_y + 1
                for j in range(new_start_y, len(board[0])):
                    if (i, j) in visited:
                        continue
                    # 寻找下个单词
                    visited.add((i, j))
                    self.dfs(board, word_set, i, j, i, j, visited, board[i][j], count, res, prefix)
                    visited.remove((i, j))

            # 不将当前字符作为单词的最后一个字符
            count -= 1
            word_set.add(word)

        # 用前缀和优化加速
        if word not in prefix:
            return

        # 如果word不代表一个单词，继续往后找(上下左右)
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if not self.is_valid(board, new_x, new_y):
                continue
            if (new_x, new_y) in visited:
                continue

            visited.add((new_x, new_y))
            self.dfs(board, word_set, new_x, new_y, start_x, start_y, visited, word + board[new_x][new_y], count, res, prefix)
            visited.remove((new_x, new_y))

    def is_valid(self, board, x, y):
        if not (0 <= x < len(board) and 0 <= y < len(board[0])):
            return False
        return True

    def get_prefix(self, words):
        prefix = {}
        for word in words:
            for i in range(len(word)):
                prefix[word[ : i + 1]] = word
                
        return prefix





# Method.2      双重DFS，
#               第一重搜索：在每个位置上搜索所有以该位置开始的单词，用Trie实现高效查找
#               第二重搜索：在每个位置上选择每个可以形成的单词，继续向下搜索，得出最终答案
class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: return the maximum nunber
    """
    def wordSearchIII(self, board, words):
        if not board or not board[0]:
            return 0

        trie = Trie()
        for w in words:
            trie.insert(w)

        return self.dfs(board, trie, set(), 0, 0)

    def dfs(self, board, trie, visited, start_i, start_j):
        n, m = len(board), len(board[0])
        word_count = 0

        for i in range(start_i, n):
            # _j = start_j + 1 if i == start_i else 0     # 不能省，否则有testcase过不了
            for j in range(start_j, m):
                if (i, j) in visited:
                    continue
                c = board[i][j]
                if c not in trie.root.children:
                    continue

                visited.add((i, j))
                word_count = max(
                    word_count,
                    self.search_word(board, i, j, trie, trie.root.children[c], visited, i, j+1),
                )
                visited.remove((i, j))
            start_j = 0

        return word_count

    def search_word(self, board, x, y, trie, node, visited, start_i, start_j):
        n, m = len(board), len(board[0])
        word_count = 0

        # a) 将当前字符串作为一个单词，count+1，从当前字符串的起点处的下一个位置开始dfs，查找新单词
        if node.is_word:
            node.is_word = False
            word_count = self.dfs(board, trie, visited, start_i, start_j) + 1
            node.is_word = True

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy
            if not (0 <= next_x < n and 0 <= next_y < m):
                continue
            if (next_x, next_y) in visited:
                continue

            c = board[next_x][next_y]
            if c not in node.children:
                continue

            visited.add((next_x, next_y))
            word_count = max(
                word_count,
                self.search_word(board, next_x, next_y, trie, node.children[c], visited, start_i, start_j),
            )
            visited.remove((next_x, next_y))

        return word_count







# 635 · Boggle Game
"""
Given a board which is a 2D matrix includes a-z and dictionary dict, find the largest collection of words 
on the board, the words can not overlap in the same position. return the size of largest collection.

Input:
["abc","def","ghi"]
{"abc","defi","gh"}
Output:     3
Explanation:
we can get the largest collection`["abc", "defi", "gh"]`
"""
# Method.1      双重DFS
# Outter dfs   to determine: given a word, what other words can be found. 
#                       This is to eliminate the overlapping.
# Inner dfs    to determine: starting from a position, what words can be found. 
#                       This is to find words
class Solution:
    """
    @param: board: a list of lists of character
    @param: words: a list of string
    @return: an integer
    """
    def boggleGame(self, board, words):
        if not board or not board[0] or not words:
            return 0
        
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        return self.outter_dfs(board, trie.root, 0, set())
    
    def outter_dfs(self, board, root, pos, visited):
        if pos == len(board) * len(board[0]):
            return 0
            
        paths = []
        i, j = divmod(pos, len(board[0]))
        
        if self.is_valid(board, root, i, j, visited):
            visited.add((i, j))
            self.inner_dfs(board, root.children[board[i][j]], i, j, visited, [(i, j)], paths)
            visited.remove((i, j))
            
        ret = self.outter_dfs(board, root, pos + 1, visited)
        
        for path in paths:
            for p_i, p_j in path:
                visited.add((p_i, p_j))

            ret = max(ret, self.outter_dfs(board, root, pos + 1, visited) + 1)

            for p_i, p_j in path:
                visited.remove((p_i, p_j))
        return ret
    
    def is_valid(self, board, node, x, y, visited):
        if not (0 <= x < len(board) and 0 <= y < len(board[0])):
            return False
        if (x, y) in visited:
            return False
        if board[x][y] not in node.children:
            return False
        return True
        
    def inner_dfs(self, board, node, x, y, visited, path, paths):
        if node.is_word:
            paths.append(list(path))
            return
        
        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy
            if not self.is_valid(board, node, next_x, next_y, visited):
                continue
            
            visited.add((next_x, next_y))
            path.append((next_x, next_y))
            self.inner_dfs(board, node.children[board[next_x][next_y]], next_x, next_y, visited, path, paths)
            path.pop()
            visited.remove((next_x, next_y))



    # Method.2      能过，但是不知道是不是能涵盖所有情况，测试数据：
    #               ["ax", "bd", "ce", "gx", "hj","ik"]
    #               {"bd","ce","hj","ik","abc","ghi"}
    def boggleGame(self, board, words):
        if not board or not board[0]:
            return 0
        
        trie = Trie()
        for word in words:
            trie.insert(word)

        self.paths = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, i, j, set([(i, j)]), trie.search(board[i][j]))

        result = 0
        for path in self.paths:
            coords = path.copy()
            count = 1
            for neighbor in self.paths:
                if neighbor.isdisjoint(coords):
                    coords.update(neighbor.copy())
                    count += 1
            result = max(count, result)

        return result

    def dfs(self, board, x, y, path, node):
        if node is None:
            return
        if node.is_word:
            self.paths.append(path.copy())

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy
            
            if not (0 <= next_x < len(board) and 0 <= next_y < len(board[0])):
                continue
            if (next_x, next_y) in path:
                continue

            path.add((next_x, next_y))
            self.dfs(board, next_x, next_y, path, node.children.get(board[next_x][next_y]))
            path.remove((next_x, next_y))