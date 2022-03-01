# 814 · Shortest Path in Undirected Graph
# 1565 · Modern Ludo I
# 803 · Shortest Distance from All Buildings
# 1364 · the minium distance
# 941 · Sliding Puzzle
# 794 · Sliding Puzzle II

#############################################################################################################
import sys
import heapq
from collections import deque



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




