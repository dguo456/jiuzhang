# 1514 · Robot Room Cleaner
"""
Given a robot cleaner in a room modeled as a grid. Each cell in the grid can be empty or blocked.
The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 
90 degrees. When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it 
stays on the current cell. Design an algorithm to clean the entire room using only the 4 given APIs.

Input:
room = [
  [1,1,1,1,1,0,1,1],
  [1,1,1,1,1,0,1,1],
  [1,0,1,1,1,1,1,1],
  [0,0,0,1,0,0,0,0],
  [1,1,1,1,1,1,1,1]
],
row = 1,
col = 3
Explanation:
All grids in the room are marked by either 0 or 1.
0 means the cell is blocked, while 1 means the cell is accessible.
The robot initially starts at the position of row=1, col=3.
From the top left corner, its position is one row below and three columns right.
"""
#class Robot:
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void


DIR_X = [-1, 0, 1, 0]
DIR_Y = [0, 1, 0, -1]
class Solution:
    """
    :type robot: Robot
    :rtype: None
    """
    def cleanRoom(self, robot):
        visited = set()
        self.dfs(robot, visited, (0, 0), 0)
    
    def dfs(self, robot, visited, pos, direction):
        if pos in visited:
            return
        visited.add(pos)
        robot.clean()

        cur_direction = direction
        for i in range(len(DIR_X)):
            if robot.move():
                new_pos = pos[0] + DIR_X[cur_direction % 4], pos[1] + DIR_Y[cur_direction % 4]
                self.dfs(robot, visited, new_pos, cur_direction)
                self.backtrack(robot)
            robot.turnRight()
            cur_direction += 1

    def backtrack(self, robot):
        robot.turnRight()
        robot.turnRight()
        robot.move()
        robot.turnRight()
        robot.turnRight()