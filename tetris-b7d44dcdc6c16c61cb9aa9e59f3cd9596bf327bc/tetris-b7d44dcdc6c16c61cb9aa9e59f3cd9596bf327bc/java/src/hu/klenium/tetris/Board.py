import os
import math
import random
from dataclasses import dataclass, field

from TetrominoDataSource import get_data

from view.SquareView import *
from view.BoardView import *


@dataclass
class Board:
    rows: int = field(init=False)
    columns: int = field(init=False)
    board: list = field(init=False)

    def __post_init__(self, rows, cols, view):
        self.rows = rows
        self.columns = cols
        self.board = SquareView[rows][cols]
        self.view = view
        self.update_view()

    @property
    def get_height(self):
        return self.rows

    @property
    def get_width(self):
        return self.columns

    def can_add_tetromino(self, tetromino, x, y):
        data = tetromino.get_polyomino_data()
        height, width = len(data), len(data[0])
        if (x < 0 or x + width > self.columns or y < 0 or y + height > self.rows):
            return False
        for i in range(height):
            for j in range(width):
                if data[i][j] and self.board[i + y][j + x]:
                    return False
        return True

    def add_tetromino(self, tetromino):
        data = tetromino.get_polyomino_data()
        height, width = len(data), len(data[0])
        x, y = tetromino.get_pos_X, tetromino.get_pos_Y
        for i in range(height):
            for j in range(width):
                if data[i][j]:
                    self.board[i + y][j + x] = data[i][j]
        self.update_view()

    def remove_rows(self):
        is_row_full = False
        for i in range(self.rows):
            is_row_full = True
            for j in range(self.columns):
                if not self.board[i][j]:
                    is_row_full = False
            if is_row_full:
                #TODO:
                pass
        self.update_view()

    def update_view(self):
        self.view.update(self.board)





@dataclass
class TetrisGame:
    rows: int = 16
    columns: int = 11
    block_size: int = 30
    is_running: bool = False
    
