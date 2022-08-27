from ast import arg
import os
import sys
import random
import pygame

from tetris import settings


class Block(pygame.sprite.Sprite):
    
    def __init__(self, img, pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img = img
        self.rect = self.img.get_rect()
        self.rect.topleft = pos
        self.row, self.col = pos

    def update(self):
        self.rect.topleft = (self.x, self.y)


class Piece(pygame.sprite.Sprite):
    