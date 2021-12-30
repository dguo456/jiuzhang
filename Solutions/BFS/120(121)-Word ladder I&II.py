# 121 · Word Ladder I  （找一条最短路径）
"""Given two words (start and end), and a dictionary, find the shortest transformation sequence from start to end, output the length of the sequence."""

from collections import deque

class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    def ladderLength(self, start, end, dict):
        dict.add(end)
        queue = deque([start])
        # distance = {start : 0}    优化,代替visited和count，且不用for当前层，而是通过distance存所有点的路径且保证是最短路径
        visited = set()
        count = 0

        while queue:
            count += 1

            for _ in range(len(queue)):
                word = queue.popleft()
                if word == end:
                    return count
                    # return distance[word]

                for next_word in self.get_next_words(word):
                    if next_word not in dict or next_word in visited:
                        continue
                    queue.append(next_word)
                    visited.add(next_word)
                    # distance[next_word] = distance[word] + 1

        return 0

    def get_next_words(self, word):
        words = []
        for i in range(len(word)):
            first_half, last_half = word[:i], word[i+1:]
            for middle in "abcdefghijklmnopqrstuvwxyz":
                if middle == word[i]:
                    continue
                words.append(first_half + middle + last_half)
        return words



# 121 · Word Ladder II  (找全部最短路径)
"""Given two words (start and end), and a dictionary, find all shortest transformation sequence(s) from start to end."""

class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: a list of lists of string
    """
    def findLadders(self, start, end, dict):
        dict.add(start)
        dict.add(end)
        distance = {}
        
        self.bfs(end, distance, dict)
        
        results = []
        self.dfs(start, end, distance, dict, [start], results)
        
        return results

    def bfs(self, start, distance, dict):
        distance[start] = 0
        queue = deque([start])

        while queue:
            word = queue.popleft()
            for next_word in self.get_next_words(word, dict):
                if next_word in distance:
                    continue
                distance[next_word] = distance[word] + 1
                queue.append(next_word)
    
    def get_next_words(self, word, dict):
        words = []
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + char + word[i+1:]
                if next_word != word and next_word in dict:
                    words.append(next_word)
        return words
                        
    def dfs(self, source, target, distance, dict, path, results):
        if source == target:
            results.append(path[:])
            return
        
        for word in self.get_next_words(source, dict):
            if distance[word] != distance[source] - 1:
                continue

            path.append(word)
            self.dfs(word, target, distance, dict, path, results)
            path.pop()