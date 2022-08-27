import os
import math
import random
from dataclasses import dataclass, field

from TetrominoDataSource import get_data

from view.SquareView import *
from view.TetrominoView import *


@dataclass
class Tetromino:
    height: int
    width: int
    curr_X: int
    curr_Y: int
    rotation: int
    parts_data: list = field(init=False, default_factory=list)

    def __post_init__(self, type, view, board):
        self.view = view
        self.board = board
        self.parts_data = TetrominoDataSource.get_data(type)
        self.set_rotation(0)

    @classmethod
    def create_at_center(cls, type, view, board):
        tetromino = cls(type, view, board)
        x = int(math.ceil((board.get_width() - tetromino.width) / 2))
        moved = tetromino.try_move(x, 0)
        if not moved:
            tetromino.view.clear()
            return None
        return tetromino

    @property
    def get_polyomino_data(self):
        return self.parts_data[self.rotation]

    @property
    def get_pos_X(self):
        return self.curr_X
    
    @property
    def get_pos_Y(self):
        return self.curr_Y

    def rotate_right(self):
        next_rotation = (self.rotation + 1) % 4
        orig_rotation = self.rotation
        self.set_rotation(next_rotation)

        can_rotate = False
        if self.can_move_to(0, 0):
            can_rotate = True
        else:
            for i in range(1, self.width):
                if not can_rotate:
                    if self.can_move_to(-i, 0):
                        self.curr_X -= i
                        can_rotate = True

        if not can_rotate:
            self.set_rotation(orig_rotation)
        else:
            self.set_rotation(next_rotation)
            self.update_view()

    def move_left(self):
        return self.try_move(-1, 0)

    def move_right(self):
        return self.try_move(1, 0)

    def move_down(self):
        return self.try_move(0, 1)

    def drop(self):
        can_move_down = True
        while can_move_down:
            can_move_down = self.move_down()

    def set_rotation(self, rotation):
        self.rotation = rotation % len(self.parts_data)
        self.height = len(self.parts_data[self.rotation])
        self.width = len(self.parts_data[self.rotation])

    def try_move(self, x, y):
        can_move = self.can_move_to(x, y)
        if can_move:
            self.curr_X += x
            self.curr_Y += y
            self.update_view()
        return can_move

    def can_move_to(self, dx, dy):
        next_X, next_Y = self.curr_X + dx, self.curr_Y + dy
        return self.board.can_add_tetromino(next_X, next_Y)

    def update_view(self):
        self.view.update(self.parts_data[self.rotation], self.curr_X, self.curr_Y)
