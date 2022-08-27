from curses import KEY_DOWN, keyname
import os
import random
import time
from turtle import position
import pygame

from collections import namedtuple
from pygame import Rect, Surface
from matris import GameOver
from scores import load_score

from tetrominoes import list_of_tetrominoes
from tetrominoes import rotate

from scores import load_score, write_score


# Set how many rows and columns we will have
ROW_COUNT = 24
COLUMN_COUNT = 10

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 30

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 5

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "Tetris"



class Matris:

    def __init__(self):
        self.matirx = dict()
        for r in range(SCREEN_HEIGHT):
            for c in range(SCREEN_WIDTH):
                self.matrix[r][c] = None

        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.6     # Move down every 600 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed) * 2

        self.level = 1
        self.score = 0
        self.lines = 0
        self.combo = 1
        self.paused = False
        self.highscore = load_score()


    def set_tetrominoes(self):
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.tetromino_position = (0, 4) if len(self.current_tetromino) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)

    def block(self, color):
        colors = {
            'blue':   (105, 105, 255),
            'yellow': (225, 242, 41),
            'pink':   (242, 41, 195),
            'green':  (22, 181, 64),
            'red':    (204, 22, 22),
            'orange': (245, 144, 12),
            'cyan':   (10, 255, 226)
        }

        end = []
        border = Surface((), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c*0.5, colors[color]) + end))
        return border

    def update(self, time_passed):
        self.need_redraw = False
        
        pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
        unpressed = lambda key: event.type == pygame.KEYUP and event.key == key

        events = pygame.event.get()
        for event in events:
            if pressed(pygame.K_p):
                # self.surface.fill((0, 0, 0))
                self.need_redraw = True
                self.paused = not self.paused
            elif event.type == pygame.QUIT:
                self.gameover(full_exit=True)
            elif pressed(pygame.K_ESCAPE):
                self.gameover()

        if self.paused:
            return self.need_redraw

        for event in events:
            if pressed(pygame.K_SPACE):
                self.instant_drop()
            elif pressed(pygame.K_UP) or pressed(pygame.K_w):
                self.request_rotation()
            elif pressed(pygame.K_LEFT) or pressed(pygame.K_a):
                self.request_movement('left')
                self.movement_keys['left'] = 1
            elif pressed(pygame.K_RIGHT) or pressed(pygame.K_d):
                self.request_movement('right')
                self.movement_keys['right'] = 1
            elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
                self.movement_keys['left'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed) * 2
            elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
                self.movement_keys['right'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed) * 2

        self.downwards_speed = self.base_downwards_speed ** (1 + self.level/10.)
        self.downwards_timer += time_passed
        downwards_speed = self.downwards_speed*0.10 if any([pygame.key.get_pressed()[pygame.KEY_DOWN], 
                                                            pygame.key.get_pressed()[pygame.K_s]]) else self.downwards_speed
        if self.downwards_timer > downwards_speed:
            if not self.request_movement('down'):
                self.lock_tetromino()
            self.downwards_timer %= downwards_speed

        if any(self.movement_keys.values()):
            self.movement_keys_timer += time_passed
        if self.movement_keys_timer > self.movement_keys_speed:
            self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed

        return self.need_redraw

    def instant_drop(self):
        amount = 0
        while self.request_movement('down'):
            amount += 1
        self.score += amount * 10
        self.lock_tetromino()

    def gameover(self, full_exit=False):
        write_score(self.score)
        if full_exit:
            exit()
        else:
            raise GameOver("Sucker!")

    def request_movement(self, direction):
        row, col = self.tetromino_position
        if direction == 'left' and self.can_move(position=(row, col-1)):
            self.tetromino_position = (row, col-1)
            self.need_redraw = True
            return self.tetromino_position
        elif direction == 'right' and self.can_move(position=(row, col+1)):
            self.tetromino_position = (row, col+1)
            self.need_redraw = True
            return self.tetromino_position
        elif direction == 'down' and self.can_move(position=(row+1, col)):
            self.tetromino_position = (row+1, col)
            self.need_redraw = True
            return self.tetromino_position
        else:
            return False

    def request_rotation(self, ):
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        r, c = self.tetromino_position
        position = (self.can_rotate(shape, (r, c)) or 
                    self.can_rotate(shape, (r, c+1)) or
                    self.can_rotate(shape, (r, c-1)) or
                    self.can_rotate(shape, (r, c+2)) or
                    self.can_rotate(shape, (r, c-2)))

        if position and self.can_move(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            self.need_redraw = True
            return self.tetromino_rotation
        else:
            return False

    def rotated(self, rotation=None):
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def lock_tetromino(self, ):
        pass

    def can_move(self, shape=None, position=None, matrix=None):
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        next_matrix = dict(self.matrix if matrix is None else matrix)
        r, c = position
        for i in range(r, r + len(shape)):
            for j in range(c, c + len(shape)):
                if (next_matrix.get((i, j), False) is False and shape[i-r][j-c] or
                    next_matrix.get((i, j), False) and shape[i-r][j-c]):
                    return False
                elif shape[i-r][j-c]:
                    next_matrix[i][j] = ('block', self.tetromino_block)

        return next_matrix

    def can_rotate(self, shape, position):
        r, c = position
        for i in range(r, r + len(shape)):
            for j in range(c, c + len(shape)):
                if self.matrix.get((i, j), False) is False and shape[i-r][j-c]:
                    return False
        return position












class Application:
    pass


class MainApplication(Application):
    
    def __init__(self, task):
        self.readyTask = task
        self.launch()

    def createFrame(self,):
        pass

    def start(self, primaryStage):
        pass


class CanvasView:
    context = None
    squareSize = 0
    contextWidth = 0
    contextHeight = 0

    def __init__(self, canvas, squareSize):
        self.squareSize = squareSize
        self.context = canvas.getGraphicsContext2D()
        self.contextWidth = canvas.getWidth()
        self.contextHeight = canvas.getHieght()

    def clear(self,):
        self.context.clearRect(0, 0, self.contextWidth, self.contextHeight)


class BoardView(CanvasView):

    def __init__(self, frame, squareSize):
        super().__init__(frame, squareSize)
    
    def update(self, data):
        self.height = len(data)
        self.width = len(data[0])
        for i in range(self.height):
            for j in range(self.width):
                if data[i][j]:
                    data[i][j].update(self.context, j, i, self.squareSize)


class TetrisGame:
    rows = 16
    columns = 11
    blockSize = 30

    isRunning = False
    random_val = random.randint()
    board = []
    fallingTetromino = None
    gravity = None
    gameFrame = None


    def TetrisGame(self):
        self.gameFrame = MainApplication.createFrame()
        self.gameFrame.setSize(self.columns * self.blockSize, self.rows * self.blockSize)
        self.gameFrame.registerEventListeners()
        view = BoardView(self.gameFrame, self.blockSize)
        board = [[]]






if __name__ == '__main__':
    MainApplication.init()

    game = TetrisGame()
    game.start()