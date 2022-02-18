# 1832 · Minimum Step
"""
There is a 1 * n chess table, indexed with 0, 1, 2 .. n - 1, every grid is colored.
And there is a chess piece on position 0, please calculate the minimum step that 
you should move it to position n-1.

Here are 3 ways to move the piece, the piece can't be moved outside of the table:
1.  Move the piece from position ii to position i + 1.
2.  Move the piece from position ii to positino i - 1.
3.  If the colors on position i and position j are same, you can 
    move the piece directly from position i to position j.
"""

# Input:
# colors = [1, 2, 3, 3, 2, 5]
# Output: 3
# Explanation: 
# In the example. you should move the piece 3 times:
# 1. Move from position 0 to position 1.
# 2. Because of the same color in position 1 and position 4, move from position 1 to position 4,
# 3. Move from position 4 to position 5.

from collections import defaultdict, deque

class Solution:
    """
    @param colors: the colors of grids
    @return: return the minimum step from position 0 to position n - 1
    """
    def minimumStep(self, colors):
        if not colors or len(colors) == 0:
            return 0

        graph = defaultdict(list)
        for index, color in enumerate(colors):
            graph[color].append(index)

        queue = deque([0])
        visited = set()
        visited.add(0)
        steps = -1

        while queue:
            steps += 1

            for _ in range(len(queue)):
                index = queue.popleft()

                if index == len(colors) - 1:
                    return steps

                for next_index in graph[colors[index]]:
                    if next_index == index:
                        continue
                    if next_index not in visited:
                        queue.append(next_index)
                        visited.add(next_index)

                graph[colors[index]] = []

                if index + 1 < len(colors) and index+1 not in visited:
                    queue.append(index+1)
                    visited.add(index+1)

                if index - 1 >= 0 and index-1 not in visited:
                    queue.append(index-1)
                    visited.add(index-1)

        return steps