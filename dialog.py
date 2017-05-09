import pygame
from constants import *


class Dialog(pygame.sprite.Sprite):
    width = 300
    height = 100
    white = (255,255,255)
    black = (0,0,0)

    text_group = []
    texts = ['Deafy hate cats', 'your help. He can not hear']
    texts2 = ['but has a sharp vision. Use your facial ', 'expression, specifically']
    texts3 = ['the extent in which you open your ', 'mouth to control Deafy\'s running']
    texts4 = ['and running and help guide him home.', ' ']
    text_group.append(texts)
    text_group.append(texts2)
    text_group.append(texts3)
    text_group.append(texts4)

    def __init__(self, diglog_index):
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 22)

        self.image = pygame.Surface([Dialog.width, Dialog.height])
        self.image.fill(Dialog.black)
        self.rect = self.image.get_rect()

        border_width = 5
        # horizontal border
        h_border = pygame.Surface([Dialog.width, border_width])
        h_border.fill(Dialog.white)
        # vertical border
        v_border = pygame.Surface([border_width, Dialog.height])
        v_border.fill(Dialog.white)

        # render current frame text
        current_text = Dialog.text_group[diglog_index]
        distance = 0
        for i in range(len(current_text)):
            textsurface = myfont.render(current_text[i], False, Dialog.white)
            self.image.blit(textsurface, (15, 15 + distance))
            distance += (i+1) * 15
        space_indicator = myfont.render('>>> Space', False, Dialog.white)
        self.image.blit(space_indicator, (200, 15 + distance))

        # add border to each edge of the image
        self.image.blit(v_border, (0,0))
        self.image.blit(v_border, (Dialog.width-border_width,0))
        self.image.blit(h_border, (0,0))
        self.image.blit(h_border, (0, Dialog.height-border_width))