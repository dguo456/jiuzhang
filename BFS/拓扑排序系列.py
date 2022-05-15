# 127 · Topological Sorting
# 605 · Sequence Reconstruction
# 615 · Course Schedule
# 616 · Course Schedule II
# 696 · Course Schedule III
# 815 · Course Schedule IV
# 305 · longest increasing path in a matrix
# 1469 · Longest Path On The Tree
# 892 · Alien Dictionary
# 1876 · Alien Dictionary(easy)

################################################################################################################

import sys
import heapq
from collections import defaultdict, deque


# 127 · Topological Sorting
"""
Given an directed graph, Find any topological order for the given graph.
Input:      graph = {0,1,2,3#1,4#2,4,5#3,4,5#4#5}
Output:     [0, 1, 2, 3, 4, 5]
"""

class DirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []

class Solution:
    """
    @param graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        indegree = self.get_indegree(graph)

        start_nodes = [node for node in graph if indegree[node] == 0]
        queue = deque(start_nodes)
        res = []

        while queue:
            node = queue.popleft()
            res.append(node)

            for neighbor in node.neighbors:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return res

    def get_indegree(self, graph):
        indegree = {x: 0 for x in graph}

        for node in graph:
            for neighbor in node.neighbors:
                indegree[neighbor] += 1

        return indegree





# 605 · Sequence Reconstruction
"""
Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. 
The org sequence is a permutation of the integers from 1 to n. Reconstruction means building a 
shortest common supersequence of the sequences in seqs (i.e., a shortest sequence so that all sequences 
in seqs are subsequences of it). Determine whether there is only one sequence that can be reconstructed 
from seqs and it is the org sequence.
"""

# Example 1:
# Input:  org = [1,2,3], seqs = [[1,2],[1,3]]
# Output: false
# Explanation:
# [1,2,3] is not the only one sequence that can be reconstructed, because [1,3,2] is also a 
# valid sequence that can be reconstructed.

# Example 2:
# Input: org = [1,2,3], seqs = [[1,2]]
# Output: false
# Explanation:
# The reconstructed sequence can only be [1,2].

# Example 3:
# Input: org = [1,2,3], seqs = [[1,2],[1,3],[2,3]]
# Output: true
# Explanation:
# The sequences [1,2], [1,3], and [2,3] can uniquely reconstruct the original sequence [1,2,3].

# Example 4:
# Input:org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
# Output:true

class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        graph = self.build_graph(seqs)
        topo_order = self.topological_sort(graph)
        return topo_order == org
            
    def build_graph(self, seqs):
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
        
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i - 1]].add(seq[i])

        return graph
    
    def get_indegrees(self, graph):
        indegrees = {node: 0 for node in graph}
        
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1
                
        return indegrees
        
    def topological_sort(self, graph):
        indegrees = self.get_indegrees(graph)
        
        start_nodes = [node for node in graph if indegrees[node] == 0]
        queue = deque(start_nodes)
        topo_order = []

        while queue:
            if len(queue) > 1:
                # 证明一定存在多种可能，不是唯一的顺序，
                return None
                
            node = queue.popleft()
            topo_order.append(node)
            
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)

        # 这里外加判断，如果长度不等，证明多点或者少点，肯定不能唯一构造出org    
        if len(topo_order) == len(graph):
            return topo_order
            
        return None





# 615 · Course Schedule
"""
There are a total of n courses you have to take, labeled from 0 to n - 1.
Before taking some courses, you need to take other courses. For example, to learn course 0, you need to learn 
course 1 first, which is expressed as [0,1]. Given the total number of courses and a list of prerequisite pairs, 
is it possible for you to finish all courses?
Example 1:
Input: n = 2, prerequisites = [[1,0]] 
Output: true

Example 2:
Input: n = 2, prerequisites = [[1,0],[0,1]] 
Output: false
"""
class Solution:
    """
    @param numCourses: a total of n courses
    @param prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        indegree = {x: 0 for x in range(numCourses)}
        implicit_graph = {x: [] for x in range(numCourses)}

        for i, j in prerequisites:
            indegree[i] += 1
            implicit_graph[j].append(i)

        start_courses = [c for c in range(numCourses) if indegree[c] == 0]
        queue = deque(start_courses)
        result = 0

        while queue:
            course = queue.popleft()
            result += 1

            for neighbor in implicit_graph[course]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return result == numCourses





# 616 · Course Schedule II
"""
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is 
expressed as a pair: [0,1]. Given the total number of courses and a list of prerequisite pairs, return the 
ordering of courses you should take to finish all courses. There may be multiple correct orders, you just 
need to return one of them. If it is impossible to finish all courses, return an empty array.
"""
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """
    def findOrder(self, numCourses, prerequisites):
        implicit_graph = {x: [] for x in range(numCourses)}
        indegree = {x: 0 for x in range(numCourses)}

        for i, j in prerequisites:
            implicit_graph[j].append(i)
            indegree[i] += 1

        start_courses = [c for c in range(numCourses) if indegree[c] == 0]
        queue = deque(start_courses)
        result = []

        while queue:
            course = queue.popleft()
            result.append(course)

            for neighbor in implicit_graph[course]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) == numCourses:
            return result

        return []





# 696 · Course Schedule III
"""
There are n different online courses numbered from 1 to n. Each course has some duration(course length) t and 
closed on dth day. A course should be taken continuously for t days and must be finished before or on the dth day. 
You will start at the 1st day. Given n online courses represented by pairs (t,d), your task is to find the maximal 
number of courses that can be taken.

Input: [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
Output: 3
"""
import heapq

class Solution:
    """
    @param courses: duration and close day of each course
    @return: the maximal number of courses that can be taken
    """
    def scheduleCourse(self, courses):
        if courses is None or len(courses) == 0:
            return 0
            
        courses.sort(key = lambda x: x[1])
        queue = []
        time = 0
        
        for i in range(len(courses)):
            if time + courses[i][0] <= courses[i][1]:
                time += courses[i][0]
                heapq.heappush(queue, -courses[i][0])
            elif queue and courses[i][0] < (-queue[0]):
                time += courses[i][0] - (-queue[0])
                heapq.heapreplace(queue, -courses[i][0])
            print(queue)
                
        return len(queue)


    # Method.2      贪心法 Greedy
    # 课程按照 deadline 排序，从左到右扫描每个课程，依次学习。如果发现学了之后超过 deadline 的，就从之前学过的课程里
    # 扔掉一个耗时最长的。因为这样可以使得其他的课程往前挪，而往前挪是没影响的。所以挑最大值这个事情就是 Heap 的事情了
    def scheduleCourse(self, courses):
        courses = sorted(courses, key=lambda x: x[1])
        curt_time = 0
        heap = []

        for duration, deadline in courses:
            curt_time += duration
            heapq.heappush(heap, -duration)
            if curt_time > deadline:
                curt_time -= (-heapq.heappop(heap))
                
        return len(heap)





# 815 · Course Schedule IV
"""
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to first take course 1, 
which is expressed as a pair: [0,1]. Given the total number of courses and a list of prerequisite pairs, 
return the number of different ways you can get all the lessons
Input:
n = 2
prerequisites = [[1,0]]
Output: 1
Explantion:
You must have class in order 0->1.
"""
class Solution:
    """
    @param n: an integer, denote the number of courses
    @param p: a list of prerequisite pairs
    @return: return an integer,denote the number of topologicalsort
    """
    # Method.1      最基础的DFS版本，一定要记得回溯。
    def topological_sort_number(self, n: int, p) -> int:
        implicit_graph = {x: set() for x in range(n)}
        indegree = {x: 0 for x in range(n)}

        for req, pre in p:
            implicit_graph[pre].add(req)
            indegree[req] += 1

        self.result = 0
        visited = set()
        self.dfs(n, implicit_graph, indegree, visited)

        return self.result

    def dfs(self, n, graph, indegree, visited):
        if len(visited) == n:
            self.result += 1
            return

        for i in range(n):
            if i not in visited and indegree[i] == 0:
                visited.add(i)
                for neighbor in graph[i]:
                    indegree[neighbor] -= 1

                self.dfs(n, graph, indegree, visited)

                for neighbor in graph[i]:
                    indegree[neighbor] += 1

                visited.remove(i)


    # Method.2      DFS + 记忆化搜索。
    # n个课程，每个课程有 已经安排/未安排 两种可能，用一个变量visited来存储每个课程的状态，最多有2^n个状态。
    # 搜索到某个状态后，继续向下搜索，其实就是转移到可能的下一个状态。一共搜索n步，纯DFS的复杂度就是 n^n
    # 但是根据上面的状态分析，通过记忆化搜索，复杂度可以降为2^n
    def topological_sort_number(self, n: int, p) -> int:
        implicit_graph = {x: set() for x in range(n)}
        indegree = {x: 0 for x in range(n)}

        for req, pre in p:
            implicit_graph[pre].add(req)
            indegree[req] += 1

        self.result = 0
        visited = [0] * n
        memo = {}
        
        return self.dfs(n, implicit_graph, indegree, 0, visited, memo)

    def dfs(self, n, graph, indegree, count, visited, memo):
        if count == n:
            return 1        # 这里返回1是因为例子：n=2, p=[[0, 1]], 如果只是return会报错

        state = tuple(visited)
        if state in memo:
            return memo[state]

        result = 0
        for i in range(n):
            if visited[i] == 0 and indegree[i] == 0:
                visited[i] = 1
                for neighbor in graph[i]:
                    indegree[neighbor] -= 1

                result += self.dfs(n, graph, indegree, count+1, visited, memo)

                for neighbor in graph[i]:
                    indegree[neighbor] += 1

                visited[i] = 0

        memo[state] = result
        return result





# 305 · longest increasing path in a matrix
"""
Given a matrix, the elements in it are all integers.
You need to find out the longest increasing path in matrix, and return the length of it.
The path can take any coordinate in the matrix as the starting point, and can move in the up, down, 
left and right directions each time, and ensure that the number on the moving route increases progressively.
You can not move out of this matrix.

Input: 
[[9,8,3],[9,2,1],[6,5,7]]
Output: 
5
Explanation: 
1 -> 2 -> 5 -> 6 -> 9
"""
from collections import deque
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class Solution:
    """
    @param matrix: A matrix
    @return: An integer.
    """
    # Method.1      算法：BFS、拓扑排序, 使用求解拓扑排序的算法，使用分层BFS，层数就是最长路径长。
    #               复杂度分析:   时间复杂度：O(n * m)    空间复杂度：O(n * m)
    def longest_increasing_path(self, matrix) -> int:
        if not matrix or not matrix[0]:
            return 0

        n, m = len(matrix), len(matrix[0])
        indegree = self.get_indegree(matrix, n, m)
        queue = deque()
        for r in range(n):
            for c in range(m):
                if indegree[(r, c)] == 0:
                    queue.append((r, c))

        path_length = 0
        while queue:
            path_length += 1
            len_q = len(queue)

            for _ in range(len_q):
                x, y = queue.popleft()
                
                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy

                    if not (0 <= next_x < n and 0 <= next_y < m):
                        continue
                    if matrix[next_x][next_y] <= matrix[x][y]:
                        continue

                    indegree[(next_x, next_y)] -= 1
                    if indegree[(next_x, next_y)] == 0:
                        queue.append((next_x, next_y))

        return path_length

    def get_indegree(self, matrix, n, m):
        indegree = {}

        for x in range(n):
            for y in range(m):
                indegree[(x, y)] = 0
                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy

                    if not (0 <= next_x < n and 0 <= next_y < m):
                        continue
                    if matrix[next_x][next_y] >= matrix[x][y]:
                        continue
                    # 这里注意是把(x, y)或(next_x, next_y)的indegree + 1
                    indegree[(x, y)] += 1

        return indegree



    # Method.2      利用dfs来扫描所有可能的路径, 用dp数组记录从(i, j)出发的最长递增路径，以此防止重复计算
    #               时间复杂度：O(nm): 因为每个点都只访问了一遍
    def longest_increasing_path(self, matrix) -> int:
        if not matrix or not matrix[0]:
            return 0

        result = 0
        n, m = len(matrix), len(matrix[0])

        # distance[i][j]表示从(i, j)点出发获得的Longest Increasing Path
        distance = [[0 for c in range(m)] for r in range(n)]
        
        for r in range(n):
            for c in range(m):
                result = max(self.dfs(matrix, n, m, distance, r, c), result)

        return result

    def dfs(self, matrix, n, m, distance, x, y):
        if distance[x][y] != 0:
            return distance[x][y]

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy

            if not (0 <= next_x < n and 0 <= next_y < m):
                continue
            if matrix[next_x][next_y] <= matrix[x][y]:
                continue

            # 从四周选一个最长的path
            distance[x][y] = max(self.dfs(matrix, n, m, distance, next_x, next_y), distance[x][y])

        distance[x][y] += 1     # 再加上当前点

        return distance[x][y]
        





# 1469 · Longest Path On The Tree
"""
Given a tree consisting of n nodes, n-1 edges. Output the distance between the two nodes with the 
furthest distance on this tree. Given three arrays of size n-1, starts, ends, and lens, indicating that 
the i-th edge is from starts[i] to ends[i] and the length is lens[i].

Input: n=5,starts=[0,0,2,2],ends=[1,2,3,4],lens=[1,2,5,6]
Output: 11
Explanation:
(3→2→4)the length of this path is `11`,as well as(4→2→3)。
"""
class Solution:
    """
    @param n: The number of nodes
    @param starts: One point of the edge
    @param ends: Another point of the edge
    @param lens: The length of the edge
    @return: Return the length of longest path on the tree.
    """
    # 两次bfs, 第一次找最远的点start, 第二次找离start最远的end
    def longestPath(self, n, starts, ends, lens):
        if n <= 1:
            return 0
        
        neighbors = {}
        for i in range(n - 1):
            start, end = starts[i], ends[i]
            distance = lens[i]
            if start not in neighbors:
                neighbors[start] = []
            if end not in neighbors:
                neighbors[end] = []
            
            neighbors[start].append((end, distance))
            neighbors[end].append((start, distance))
        
        # return: 距离root最远的点，最远的点离root的距离
        start, _ = self.bfs(0, neighbors)
        end, answer = self.bfs(start, neighbors)
        
        return answer
        
    def bfs(self, root, neighbors):
        queue = deque([root])
        distance = {root: 0}
        
        max_node = -1 
        max_distance = 0
        
        while queue:
            node = queue.popleft()
            
            if max_distance < distance[node]:
                max_distance = distance[node]
                max_node = node 
                
            for neighbor, edge_length in neighbors[node]:
                if neighbor in distance:
                    continue
                queue.append(neighbor)
                distance[neighbor] = distance[node] + edge_length
        
        return max_node, max_distance







# 892 · Alien Dictionary
"""
There is a new alien language which uses the latin alphabet. 
However, the order among letters are unknown to you. You receive a list of non-empty words 
from the dictionary, where words are sorted lexicographically by the rules of this new language. 
Derive the order of letters in this language.
1. You may assume all letters are in lowercase.
2. The dictionary is invalid, if string a is prefix of string b and b is appear before a.
3. If the order is invalid, return an empty string.
4. There may be multiple valid order of letters, return the smallest in normal lexicographical order.
5. The letters in one string are of the same rank by default and are sorted in Human dictionary order.

Input: ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
"""
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        Graph = {ch: [] for word in words for ch in word}
        indegree = {ch: 0 for word in words for ch in word}
        
        for pos in range(len(words)-1):
            for i in range(min(len(words[pos]), len(words[pos+1]))):
                prev, curr = words[pos][i], words[pos+1][i]
                if prev != curr:
                    Graph[prev].append(curr)
                    indegree[curr] += 1
                    break
            if prev == curr and len(words[pos]) > len(words[pos+1]):
                return ""
                
        starting_node = [ch for ch in indegree if indegree[ch] == 0]
        queue = deque(starting_node)
        order = []
        
        while queue:
            for _ in range(len(queue)):
                ch = queue.popleft()
                order.append(ch)
                for neighbor in Graph[ch]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
                        
        if len(order) != len(indegree):
            return ""
        return ''.join(order)







# 1876 · Alien Dictionary(easy)
"""
In an alien language, surprisingly they also use english lowercase letters, 
but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.
Given a sequence of words written in the alien language, and the order of the alphabet, 
return true if and only if the given words are sorted lexicographically in this alien language.
Otherwise, it returns false.
"""
class Solution:
    """
    @param words: the array of string means the list of words
    @param order: a string indicate the order of letters
    @return: return true or false
    """
    def is_alien_sorted(self, words, order: str) -> bool:
        index = {value : index for index, value in enumerate(order)}
        return words == sorted(words, key=lambda w: [index[x] for x in w])
