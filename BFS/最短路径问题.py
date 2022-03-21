# 121 · Word Ladder I&II  （找一条(全部)最短路径)
# 796 · Open the Lock
# 814 · Shortest Path in Undirected Graph
# 1565 · Modern Ludo I
# 803 · Shortest Distance from All Buildings
# 1364 · the minium distance
# 941 · Sliding Puzzle
# 794 · Sliding Puzzle II
# 950 · Sliding Puzzle III
# 611 · Knight Shortest Path
# 630 · Knight Shortest Path II
# 787 · The Maze
# 788 · The Maze II
# 789 · The Maze III

#############################################################################################################
import sys
import heapq
from collections import defaultdict, deque


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






# 796 · Open the Lock
"""
You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', 
'5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', 
or '0' to be '9'. Each move consists of turning one wheel one slot.
The lock initially starts at '0000', a string representing the state of the 4 wheels.
You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels 
of the lock will stop turning and you will be unable to open it.
Given a target representing the value of the wheels that will unlock the lock, return the minimum total number 
of turns required to open the lock, or -1 if it is impossible.

Given deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Return 6

Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".
"""

class Solution:
    """
    @param deadends: the list of deadends
    @param target: the value of the wheels that will unlock the lock
    @return: the minimum total number of turns 
    """
    # 难点在于这是一个Implicit Graph，将其转化为：简单无向图知道起点和终点求最短路径问题，
    # 所以使用BFS解题，思路和Word Ladder非常类似，同 611
    def openLock(self, deadends, target):
        deadends_set = set(deadends)
        start = "0000"

        if start in deadends_set:
            return -1 

        queue = deque([start])
        distance = {"0000": 0}

        while queue:
            len_q = len(queue)
            
            for _ in range(len_q):
                code = queue.popleft()
                
                if code == target:
                    return distance[code]
                
                for next_code in self.get_next_codes(code, deadends_set):
                    if next_code in distance:
                        continue
                    queue.append(next_code)
                    distance[next_code] = distance[code] + 1 
                    
        return -1 
        
        
    def get_next_codes(self, code, deadends):
        next_codes = [] 
        
        for i in range(len(code)):
            left, mid, right = code[:i], code[i], code[i + 1:]
            for digit in [(int(mid) + 1) % 10, (int(mid) - 1) % 10]:
                next_code = left + str(digit) + right 
                
                if next_code in deadends:
                    continue 
                
                next_codes.append(next_code)
                
        return next_codes






# 814 · Shortest Path in Undirected Graph
"""
Given an undirected graph in which each edge's length is 1, and two nodes from the graph. 
Return the length of the shortest path between two given nodes.
"""
class UndirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []

class Solution:
    """
    @param graph: a list of Undirected graph node
    @param A: nodeA
    @param B: nodeB
    @return:  the length of the shortest path
    """
    def shortestPath(self, graph, A, B):
        if A == B:
            return 0

        queue = deque([A])
        visited = set([A])
        path = 0

        while queue:
            path += 1
            len_q = len(queue)

            for _ in range(len_q):
                node = queue.popleft()
                visited.add(node)

                for neighbor in node.neighbors:
                    if neighbor == B:
                        return path
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)

        return -1

    # 另一种优化方法，建立distance用来存储所有中间路径，同时去重
    def shortestPath(self, graph, A, B):
        if A == B:
            return 0

        queue = deque([A])
        distance = {A : 0}

        while queue:
            node = queue.popleft()
            if node == B:
                return distance[node]
            for neighbor in node.neighbors:
                if neighbor in distance:
                    continue
                queue.append(neighbor)
                distance[neighbor] = distance[node] + 1

        return -1





# 1565 · Modern Ludo I
"""
There is a one-dimensional board with a starting point on the far left side of the board and an end point 
on the far right side of the board. There are several positions on the board that are connected to other positions,
ie if A is connected to B, then when chess falls at position A, you can choose whether to give up to throw dice 
and move the chess from A to B directly. And the connection is one way, which means that the chess cannot move 
from B to A. Now given the length and connections of the board, and you have a six-sided dice(1-6), output the 
minimum steps to reach the end point.

Input: length = 15 and connections = [[2, 8],
[6, 9]]
Output: 2
Explanation: 
1->6 (dice)
6->9 (for free)
9->15(dice)
"""
class Solution:
    """
    @param length: the length of board
    @param connections: the connections of the positions
    @return: the minimum steps to reach the end
    """
    # Method.1      两个队列交替，因为不用popleft，所以每层需要一个nextQueue来进行替换当前层的queue，
    #               因为不用每次popleft，不会有元素左移一位而增加时间复杂度的问题。如按照标准的popleft操作，则要用deque
    def modern_ludo(self, length, connections) -> int:
        graph = self.build_graph(length, connections)

        # queue = collections.deque([1])
        queue = [1]
        distance = {1: 0}

        while queue:
            next_queue = []

            for node in queue:
                for connected_node in graph[node]:
                    if connected_node in distance:
                        continue
                    queue.append(connected_node)
                    distance[connected_node] = distance[node]

            for node in queue:
                for next_node in range(node + 1, min(node + 7, length + 1)):
                    if next_node in distance:
                        continue
                    next_queue.append(next_node)
                    distance[next_node] = distance[node] + 1

            queue = next_queue

        return distance[length]


    # Method.2      SPFA解法，shortest path faster algorithm
    def modernLudo(self, length, connections):
        from collections import deque
        graph = self.build_graph(length, connections)
        
        queue = deque([1])
        distance = {
            i: float('inf')
            for i in range(1, length + 1)
        }
        distance[1] = 0

        while queue:
            node = queue.popleft()

            for next_node in graph[node]:
                if distance[next_node] > distance[node]:
                    distance[next_node] = distance[node]
                    queue.append(next_node)

            for next_node in range(node + 1, min(node + 7, length + 1)):
                if distance[next_node] > distance[node] + 1:
                    distance[next_node] = distance[node] + 1
                    queue.append(next_node)

        return distance[length]


    # Method.3      SPFA + 堆优化
    def modernLudo(self, length, connections):
        graph = self.build_graph(length, connections)
        queue = [(0, 1)]
        distance = {
            i: float('inf')
            for i in range(1, length + 1)
        }
        distance[1] = 0

        while queue:
            dist, node = heapq.heappop(queue)

            for next_node in graph[node]:
                if distance[next_node] > dist:
                    distance[next_node] = dist
                    heapq.heappush(queue, (dist, next_node))

            for next_node in range(node + 1, min(node + 7, length + 1)):
                if distance[next_node] > dist + 1:
                    distance[next_node] = dist + 1
                    heapq.heappush(queue, (dist + 1, next_node))

        return distance[length]


    def build_graph(self, length, connections):
        graph = {i: set() for i in range(1, length + 1)}

        for x, y in connections:
            graph[x].add(y)

        return graph





# 803 · Shortest Distance from All Buildings
"""
You want to build a house on an empty land which reaches all buildings in the shortest amount of distance. 
You can only move up, down, left and right. You are given a 2D grid of values 0, 1 or 2, where:

Each 0 marks an empty land which you can pass by freely.
Each 1 marks a building which you cannot pass through.
Each 2 marks an obstacle which you cannot pass through.

Input: [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]
Output: 7
Explanation:
In this example, there are three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2).
1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0
The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal.
"""
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Solution:
    """
    @param grid: the 2D grid
    @return: the shortest distance
    """
    def shortest_distance(self, grid) -> int:
        if not grid or not grid[0]:
            return None

        min_distance = sys.maxsize
        total_of_buildings = 0

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    total_of_buildings += 1

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 0:
                    buildings, distance = self.bfs(grid, r, c)
                    if buildings != total_of_buildings:
                        continue
                    min_distance = min(distance, min_distance)

        if min_distance == sys.maxsize:
            return -1
        return min_distance

    def bfs(self, grid, x, y):
        queue = deque([(x, y)])
        visited = set((x, y))
        total_distance = 0
        num_of_buildings = 0

        distance = 0
        while queue:
            len_q = len(queue)
            distance += 1

            for _ in range(len_q):
                x, y = queue.popleft()
                # visited.add((x, y))
                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not (0 <= next_x < len(grid) and 0 <= next_y < len(grid[0])):
                        continue
                    if (next_x, next_y) in visited:
                        continue
                    if grid[next_x][next_y] == 2:
                        continue
                    # 题目要求必须能reach到所有的building，因此需要统计被访问到的building个数
                    if grid[next_x][next_y] == 1:
                        total_distance += distance
                        num_of_buildings += 1
                    if grid[next_x][next_y] == 0:
                        queue.append((next_x, next_y))
                    # 这里注意不仅0需要标记，1也需要被标记
                    visited.add((next_x, next_y))

        return num_of_buildings, total_distance





# 1364 · the minium distance
"""
You are now given a two-dimensional tabular graph, in which each grid contains a integer num.
If num is - 2, it means this grid is the starting grid. If num is - 3, it means this grid is the ending grid. 
If num is - 1, it means this grid has an obstacle on it and you can't move to it. If num is a positive number or 0，
you can walk normally on it.
In each move you can travel from one grid to another if and only if they're next to each other or 
they contain the same positive number num. The cost of each move is 1.
Now you are asked to find the lowest cost of travelling from the starting grid to the ending grid. 
If the ending grid could not be reached, print -1.

Example
Input:[[1,0,-1,1],[-2,0,1,-3],[2,2,0,0]]
Output:3
In this example,you can reach the ending grid through these moves:
First, move up from the starting grid to the grid that contains the number 1. Second, move to the grid 
with the same number at the top right.
Finally, move down to the ending grid. There are three moves in total, so the minimun cost will be 3.
"""
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class Solution:
    """
    @param maze_map: a 2D grid
    @return: return the minium distance
    """
    # 会超时，但是代码应该是正确的
    def get_min_distance(self, maze_map) -> int:
        graph = self.build_graph(maze_map)

        queue = deque([])
        visited = set()
        visited_num = set()

        for r in range(len(maze_map)):
            for c in range(len(maze_map[0])):
                if maze_map[r][c] == -2:
                    queue.append((r, c))

        distance = 0
        while queue:
            distance += 1
            len_q = len(queue)

            for _ in range(len_q):
                x, y = queue.popleft()

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy

                    if not (0 <= next_x < len(maze_map) and 0 <= next_y < len(maze_map[0])):
                        continue
                    if (next_x, next_y) in visited:
                        continue

                    if maze_map[next_x][next_y] == -3:
                        return distance

                    if maze_map[next_x][next_y] >= 0:
                        queue.append((next_x, next_y))
                        visited.add((next_x, next_y))

                if maze_map[x][y] in visited_num:
                    continue
                
                visited_num.add(maze_map[x][y])
                if maze_map[x][y] in graph:
                    for connected_node in graph[maze_map[x][y]]:
                        # print(connected_node)
                        if connected_node in visited:
                            continue
                        queue.append(connected_node)
                        visited.add(connected_node)

                    del graph[maze_map[x][y]]
    
        return -1

    def build_graph(self, maze_map):
        graph = {}

        for r in range(len(maze_map)):
            for c in range(len(maze_map[0])):
                if maze_map[r][c] > 0:
                    if maze_map[r][c] not in graph:
                        graph[maze_map[r][c]] = set([(r, c)])
                    graph[maze_map[r][c]].add((r, c))

        return graph






# 941 · Sliding Puzzle
"""
On a 2x3 board, there are 5 tiles represented by the integers 1 through 5, and an empty square represented by 0.
A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.
The state of the board is solved if and only if the board is [[1,2,3],[4,5,0]].
Given a puzzle board, return the least number of moves required so that the state of the board is solved. 
If it is impossible for the state of the board to be solved, return -1.

Given board = `[[1,2,3],[4,0,5]]`, return `1`.
Explanation: 
Swap the 0 and the 5 in one move.
"""

class Solution:
    """
    @param board: the given board
    @return:  the least number of moves required so that the state of the board is solved
    """
    # 对标794, 标准 BFS，但是难点并不在 BFS 的实现，而是需要确定好存入queue里到底是什么
    # 注意先转化为string 不然list在visited 的set中不是hashable的
    def slidingPuzzle(self, board):
        final_state = [[1, 2, 3], [4, 5, 0]]
        source = self.matrix_to_string(board)
        target = self.matrix_to_string(final_state)

        queue = deque([source])
        distance = {source: 0}
        
        while queue:
            curr = queue.popleft()
            if curr == target:
                return distance[curr]
            for next in self.get_next(curr):
                if next in distance:
                    continue
                queue.append(next)
                distance[next] = distance[curr] + 1
        
        return -1

    def get_next(self, state):
        states = []
        direction = ((0, 1), (1, 0), (-1, 0), (0, -1))
        
        zero_index = state.find('0')
        x, y = zero_index // 3, zero_index % 3
        
        for i in range(4):
            x_, y_ = x + direction[i][0], y + direction[i][1]
            if 0 <= x_ < 2 and 0 <= y_ < 3:
                next_state = list(state)
                next_state[x * 3 + y] = next_state[x_ * 3 + y_]
                next_state[x_ * 3 + y_] = '0'
                states.append("".join(next_state))
        return states
        
    def matrix_to_string(self, state):
        str_list = []
        for i in range(2):
            for j in range(3):
                str_list.append(str(state[i][j]))
        return "".join(str_list)





# 794 · Sliding Puzzle II               (same code with Sliding Puzzle)
"""
On a 3x3 board, there are 8 tiles represented by the integers 1 through 8, and an empty square represented by 0.
A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.
Given an initial state of the puzzle board and final state, return the least number of moves required so that 
the initial state to final state. If it is impossible to move from initial state to final state, return -1.
"""
class Solution:
    """
    @param init_state: the initial state of chessboard
    @param final_state: the final state of chessboard
    @return: return an integer, denote the number of minimum moving
    """
    def min_move_step(self, init_state, final_state) -> int:
        source = self.matrix_to_string(init_state)
        target = self.matrix_to_string(final_state)

        queue = deque([source])
        distance = {source: 0}

        while queue:
            curr_state = queue.popleft()
            if curr_state == target:
                return distance[curr_state]

            for next_state in self.get_next_state(curr_state):
                if next_state in distance:
                    continue
                queue.append(next_state)
                distance[next_state] = distance[curr_state] + 1

        return -1

    def matrix_to_string(self, matrix):
        string_list = []
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                string_list.append(str(matrix[r][c]))

        return "".join(string_list)

    def get_next_state(self, state):
        states = []

        zero_index = state.find('0')
        x, y = zero_index // 3, zero_index % 3

        for dx, dy in DIRECTIONS:
            next_x, next_y = x + dx, y + dy

            if not (0 <= next_x < 3 and 0 <= next_y < 3):
                continue
            next_state = list(state)
            next_state[x * 3 + y] = next_state[next_x * 3 + next_y]
            next_state[next_x * 3 + next_y] = '0'
            states.append("".join(next_state))

        return states





# 950 · Sliding Puzzle III          (代码和II完全一样)
"""
Given a 3 x 3 matrix, the number is 1~9, among which 8 squares have numbers, 1~8, and one is null 
(indicated by 0), asking if the corresponding number can be put on the corresponding label In the grid 
(spaces can only be swapped with up, down, left, and right positions), if it can output "YES", 
otherwise it outputs "NO".
"""






# 611 · Knight Shortest Path
"""
Given a knight in a chessboard (a binary matrix with 0 as empty and 1 as barrier) with a source position, 
find the shortest path to a destination position, return the length of the route.
Return -1 if destination cannot be reached.
source and destination must be empty.
Knight can not enter the barrier.
Path length refers to the number of steps the knight takes.
If the knight is at (x, y), he can get to the following positions in one step:
(x + 1, y + 2)
(x + 1, y - 2)
(x - 1, y + 2)
(x - 1, y - 2)
(x + 2, y + 1)
(x + 2, y - 1)
(x - 2, y + 1)
(x - 2, y - 1)

Input:
[[0,0,0],
 [0,0,0],
 [0,0,0]]
source = [2, 0] destination = [2, 2] 
Output: 2
Explanation:
[2,0]->[0,1]->[2,2]
"""

# Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    def shortestPath(self, grid, source, destination):
        if not grid or not grid[0]:
            return -1
        if source == destination:
            return -1

        moves = [
                    (1, 2, 0, 1), (-1, 2, 0, 1), 
                    (1, -2, 0, -1), (-1, -2, 0, -1), 
                    (2, 1, 1, 0), (2, -1, 1, 0), 
                    (-2, 1, -1, 0), (-2, -1, -1, 0)
                ]

        queue = deque([(source.x, source.y)])
        visited = set()

        step = 0
        while queue:
            l_queue = len(queue)

            for _ in range(l_queue):
                x, y = queue.popleft()
                visited.add((x, y))

                if (x, y) == (destination.x, destination.y):
                    return step

                for dx, dy, bx, by in moves:
                    next_x, next_y = x + dx, y + dy
                    if not self.isValid(grid, next_x, next_y, visited):
                        continue
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))

            step += 1

        return -1

    def isValid(self, grid, x, y, visited):
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] == 1:
            return False
        if (x, y) in visited:
            return False
        return True






# 630 · Knight Shortest Path II
"""
Given a knight in a chessboard n * m (a binary matrix with 0 as empty and 1 as barrier). the knight 
initialze position is (0, 0) and he wants to reach position (n - 1, m - 1), Knight can only be from left to right. 
Find the shortest path to the destination position, return the length of the route. 
Return -1 if knight can not reached.
If the knight is at (x, y), he can get to the following positions in one step:

(x + 1, y + 2)
(x - 1, y + 2)
(x + 2, y + 1)
(x - 2, y + 1)

Input:      [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Output:     3
Explanation:
[0,0]->[2,1]->[0,2]->[2,3]
"""
DIRECTIONS = [(1, 2), (-1, 2), (2, 1), (-2, 1)]

class Solution:
    """
    @param grid: a chessboard included 0 and 1
    @return: the shortest path
    """
    def shortestPath2(self, grid):
        if not grid or not grid[0]:
            return -1

        n, m = len(grid), len(grid[0])
        queue = deque([(0, 0)])
        visited = set()

        path = 0
        while queue:
            l_queue = len(queue)

            for _ in range(l_queue):
                x, y = queue.popleft()
                visited.add((x, y))

                if (x, y) == (n-1, m-1):
                    return path

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy
                    if not self.isValid(grid, next_x, next_y, visited):
                        continue
                    queue.append((next_x,next_y))
                    visited.add((next_x, next_y))

            path += 1
            
        return -1

    def isValid(self, grid, x, y, visited):
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] == 1:
            return False
        if (x, y) in visited:
            return False
        return True





# 787 · The Maze
"""
There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by 
rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, 
it could choose the next direction. Given the ball's start position, the destination and the maze, 
determine whether the ball could stop at the destination.
The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. 
You may assume that the borders of the maze are all walls. The start and destination coordinates 
are represented by row and column indexes.

Example 1:
Input:
map = 
[
 [0,0,1,0,0],
 [0,0,0,0,0],
 [0,0,0,1,0],
 [1,1,0,1,1],
 [0,0,0,0,0]
]
start = [0,4]   end = [3,2]
Output:         false

Example 2:
Input:
map = 
[[0,0,1,0,0],
 [0,0,0,0,0],
 [0,0,0,1,0],
 [1,1,0,1,1],
 [0,0,0,0,0]
]
start = [0,4]   end = [4,4]
Output:         true
"""
class Solution:
    """
    @param maze: the maze
    @param start: the start
    @param destination: the destination
    @return: whether the ball could stop at the destination
    """
    def has_path(self, maze, start, destination) -> bool:
        if not maze or not maze[0]:
            return False

        queue = deque([start])
        visited = set(start)

        while queue:
            len_q = len(queue)

            for _ in range(len_q):
                x, y = queue.popleft()
                # maze[x][y] = 2         可以省去visited

                if x == destination[0] and y == destination[1]:
                    return True

                for dx, dy in DIRECTIONS:
                    next_x, next_y = x + dx, y + dy

                    # 本体精髓，用一个while循环来模拟球撞停之前的轨迹，循环刚一结束马上接着还要回溯一格
                    while (0 <= next_x < len(maze) and 0 <= next_y < len(maze[0])) and maze[next_x][next_y] == 0:
                        next_x += dx
                        next_y += dy

                    next_x -= dx
                    next_y -= dy

                    if maze[next_x][next_y] == 0 and (next_x, next_y) not in visited:
                        queue.append((next_x, next_y))
                        visited.add((next_x, next_y))

        return False





# 788 · The Maze II
"""
There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, 
down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose 
the next direction. Given the ball's start position, the destination and the maze, find the shortest distance 
for the ball to stop at the destination. The distance is defined by the number of empty spaces traveled by the 
ball from the start position (excluded) to the destination (included). If the ball cannot stop at the 
destination, return -1.
The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. You may assume 
that the borders of the maze are all walls. The start and destination coordinates are represented by row and 
column indexes.

Example 1:
	Input:  
	(rowStart, colStart) = (0,4)
	(rowDest, colDest)= (4,4)
	0 0 1 0 0
	0 0 0 0 0
	0 0 0 1 0
	1 1 0 1 1
	0 0 0 0 0

	Output:  12
	
	Explanation:
	(0,4)->(0,3)->(1,3)->(1,2)->(1,1)->(1,0)->(2,0)->(2,1)->(2,2)->(3,2)->(4,2)->(4,3)->(4,4)

Example 2:
	Input:
	(rowStart, colStart) = (0,4)
	(rowDest, colDest)= (0,0)
	0 0 1 0 0
	0 0 0 0 0
	0 0 0 1 0
	1 1 0 1 1
	0 0 0 0 0

	Output:  6
	
	Explanation:
	(0,4)->(0,3)->(1,3)->(1,2)->(1,1)->(1,0)->(0,0)
"""

class Solution:
    """
    @param maze: the maze
    @param start: the start
    @param destination: the destination
    @return: the shortest distance for the ball to stop at the destination
    """
    def shortest_distance(self, maze, start, destination) -> int:
        if not maze or not maze[0]:
            return False

        queue = deque([(start[0], start[1], 0)])
        visited = [[float('inf') for _ in range(len(maze[0]))] for _ in range(len(maze))]
        visited[start[0]][start[1]] = 0

        while queue:
            x, y, prev_dist = queue.popleft()

            for dx, dy in DIRECTIONS:
                next_x, next_y, count = x + dx, y + dy, prev_dist + 1       # 这里注意下一层需要 +1

                while (0 <= next_x < len(maze) and 0 <= next_y < len(maze[0])) and maze[next_x][next_y] == 0:
                    next_x += dx
                    next_y += dy
                    count += 1

                next_x -= dx    # 这里主要回溯一步
                next_y -= dy
                count -= 1

                if count < visited[next_x][next_y]:     # 说明存在更快到达该点的路径，就更新，入queue，相当于选路走
                    visited[next_x][next_y] = count     # 仔细想其实这里visited兼顾了去重的作用
                    queue.append((next_x, next_y, count))

        if visited[destination[0]][destination[1]] == float('inf'):
            return -1       # 如果终点没有被更新，说明此路不通
        return visited[destination[0]][destination[1]]





# 789 · The Maze III
"""
There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling 
up (u), down (d), left (l) or right (r), but it won't stop rolling until hitting a wall. When the ball stops, 
it could choose the next direction. There is also a hole in this maze. The ball will drop into the hole if 
it rolls on to the hole.
Given the position of the ball, the position of the hole and the maze, find out how the ball falls into 
the hole by moving the shortest distance. The distance is defined by the number of empty spaces the ball 
passes from the starting position (excluded) to the hole (included). Use "u", "d", "l" and "r" to output 
the direction of movement. Since there may be several different shortest paths, you should output the 
shortest method in alphabetical order. If the ball doesn't go into the hole, output "impossible".
The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. You may assume 
that the borders of the maze are all walls. The ball and the hole coordinates are represented by row and 
column indexes. There is only one ball and one hole in the maze.

Input:
[[0,0,0,0,0],[1,1,0,0,1],[0,0,0,0,0],[0,1,0,0,1],[0,1,0,0,0]]
[4,3]
[0,1]

Output: "lul"
"""
DIRECTIONS_HASH = {
    'u': (-1, 0), 
    'd': (1, 0), 
    'l': (0, -1), 
    'r': (0, 1)
}
class Solution:
    """
    @param maze: the maze
    @param ball: the ball position
    @param hole: the hole position
    @return: the lexicographically smallest way
    """
    def find_shortest_way(self, maze, ball, hole) -> str:
        if not maze or not maze[0]:
            return "impossible"
        if not ball or not hole:
            return "impossible"

        hole = (hole[0], hole[1])

        queue = deque([(ball[0], ball[1])])
        # queue = [(0, '', ball[0], ball[1])]               如果把queue替换成PriorityQueue
        distance = {(ball[0], ball[1]): (0, '')}

        while queue:
            x, y = queue.popleft()
            dist, path = distance[(x, y)]
            # dist, path, x, y = heapq.heappop(queue)

            for direction in DIRECTIONS_HASH:
                if path and path[-1] == direction:
                    continue
                
                dx, dy = DIRECTIONS_HASH[direction]
                next_x, next_y = x + dx, y + dy

                while (0 <= next_x < len(maze) and 0 <= next_y < len(maze[0])) \
                    and maze[next_x][next_y] == 0 and (next_x, next_y) != hole:
                    next_x += dx
                    next_y += dy

                if (next_x, next_y) != hole:
                    next_x -= dx
                    next_y -= dy

                next_dist = dist + abs(next_x - x) + abs(next_y - y)
                new_path = path + direction

                # TODO, 这里注意不能是 next_dist >= distance[(next_x, next_y)][0]做判断
                if (next_x, next_y) in distance and (next_dist, new_path) >= distance[(next_x, next_y)]:
                    continue
                
                queue.append((next_x, next_y))
                # heapq.heappush(queue, (next_dist, new_path, next_x, next_y))
                distance[(next_x, next_y)] = (next_dist, new_path)

        if hole in distance:
            return distance[hole][1]

        return "impossible"






# 1685 · The mazeIV
"""
Give you a map where 'S' indicates the starting point and 'T' indicates the ending point. '#' means that 
the wall is unable to pass, and '.' means that the road can take a minute to pass. Please find the minimum 
time it takes to get from the start point to the end point. If the end point cannot be reached, output -1.

input:map=[['S','.'],['#','T']]
output:t=2
"""
class Solution:
    """
    @param maps: 
    @return: 
    """
    def the_maze_i_v(self, maps) -> int:
        if not maps or not maps[0]:
            return -1

        queue = deque()
        distance = {}

        start = end = (0, 0)
        for r in range(len(maps)):
            for c in range(len(maps[0])):
                if maps[r][c] == 'S':
                    start = (r, c)
                    queue.append(start)
                    distance[start] = 0
                if maps[r][c] == 'T':
                    end = (r, c)

        while queue:
            x, y = queue.popleft()

            if x == end[0] and y == end[1]:
                return distance[(x, y)]

            for dx, dy in DIRECTIONS:
                next_x, next_y = x + dx, y + dy

                if not (0 <= next_x < len(maps) and 0 <= next_y < len(maps[0])):
                    continue
                if maps[next_x][next_y] == '#':
                    continue

                if (next_x, next_y) in distance and distance[(next_x, next_y)] <= distance[(x, y)] + 1:
                    continue

                queue.append((next_x, next_y))
                distance[(next_x, next_y)] = distance[(x, y)] + 1

        if end in distance:
            return distance[end]
        
        return -1