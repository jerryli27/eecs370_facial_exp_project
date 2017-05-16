"""
This file contains some utility function for showing text.
"""

import pygame
from constants import *


class Text:
    def __init__(self, color = WHITE, font='Comic Sans MS', size=22):
        # Start PyGame Font
        pygame.font.init()
        self.font = pygame.font.SysFont(font, size)
        self.color = color

    def get_text_surface(self, text, antialias=True):
        return self.font.render(text, antialias, self.color)

    def get_text_size(self, text):
        return self.font.size(text)  # (width, height)

    def blit_text_centered_at(self, text, pos, screen, antialias=True):
        size = self.get_text_size(text)
        x = pos[0] - (size[0] / 2.)
        y = pos[1] - (size[1] / 2.)
        coords = (x, y)
        surface = self.get_text_surface(text, antialias=antialias)
        screen.blit(surface, coords)

    def blit_text_top_left_corner_at(self, text, pos, screen, antialias=True):
        x = pos[0]
        y = pos[1]
        coords = (x, y)
        surface = self.get_text_surface(text, antialias=antialias)
        screen.blit(surface, coords)

    def blit_text_bottom_left_corner_at(self, text, pos, screen, antialias=True):
        size = self.get_text_size(text)
        x = pos[0]
        y = pos[1] - size[1]
        coords = (x, y)
        surface = self.get_text_surface(text, antialias=antialias)
        screen.blit(surface, coords)

    def change_color(self, color):
        self.color = color