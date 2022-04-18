# 1891 · Travel Plan
"""
There are n cities, and the adjacency matrix arr represents the distance between any two cities.
arr[i][j] represents the distance from city i to city j .Alice made a travel plan on the weekend. 
She started from city 0, then she traveled other cities 1 ~ n-1, and finally returned to city 0. 
Alice wants to know the minimum distance she needs to walk to complete the travel plan. 
Return this minimum distance. Except for city 0, every city can only pass once, and city 0 can only 
be the starting point and destination. Alice can't pass city 0 during travel.

Input:
[[0,1,2],[1,0,2],[2,1,0]]
Output:
4
Explanation:
There are two possible plans.
The first, city 0-> city 1-> city 2-> city 0, cost = 5.
The second, city 0-> city 2-> city 1-> city 0, cost = 4.
Return 4
"""
import sys

class Solution:
    """
    @param arr: the distance between any two cities
    @return: the minimum distance Alice needs to walk to complete the travel plan
    """
    def travel_plan(self, arr) -> int:
        if not arr:
            return 0

        visited = set([0])
        self.result = sys.maxsize
        self.dfs(arr, visited, 0, 0)
        return self.result

    def dfs(self, arr, visited, cost, city):
        if len(visited) == len(arr):
            self.result = min(self.result, cost + arr[city][0])
            return

        if cost > self.result:
            return

        # pruning with memo
        for i in range(1, len(arr)):
            if i in visited:
                continue

            visited.add(i)
            self.dfs(arr, visited, cost + arr[city][i], i)
            visited.remove(i)