import pygame
from constants import *


TUTORIAL_TEXT = [['To Move:', 'Move your head', 'left, right, up, down'],
                 ['To Fire a Normal Bullet: ', 'Open your mouth'],
                 ['To Fire an Advanced: ', 'Blink your eyes'],
                 ['Every bullet firing has', 'cool down time'],
                 ['Tutorial vague enough?', 'You will get it once you start'],
                 ['Ready? ', 'Go!']]

def get_restart_text(winner='', game_over=False):
    result = 'Congrats! ' + winner + (' wins!' if game_over else ' wins this round!') if winner != 'Draw' else 'Draw! This is rare...'
    text = [[result, 'Press C for calibration', 'or press SPACE to continue']]
    return text


class Dialog(pygame.sprite.Sprite):
    width = 230
    height = 130
    white = (255,255,255)
    black = (0,0,0)


    def __init__(self, text_group, pos=GAME_SCREEN_RECT.midbottom, is_active=True):
        pygame.sprite.Sprite.__init__(self, self.containers)
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 22)

        self.image = pygame.Surface([Dialog.width, Dialog.height])
        self.image.fill(Dialog.black)
        self.pos = pos
        self.rect = self.image.get_rect(midbottom=pos)
        self.text_group = text_group
        self.dialog_index = 0
        self.is_active = is_active
        self.dialog_count = len(text_group)

        if self.is_active == False or self.dialog_index >= self.dialog_count:
            return

        border_width = 5
        # horizontal border
        h_border = pygame.Surface([Dialog.width, border_width])
        h_border.fill(Dialog.white)
        # vertical border
        v_border = pygame.Surface([border_width, Dialog.height])
        v_border.fill(Dialog.white)

        # render current frame text
        if self.dialog_index < self.dialog_count:
            current_text = self.text_group[self.dialog_index]
            distance = 0
            for i in range(len(current_text)):
                textsurface = myfont.render(current_text[i], False, Dialog.white)
                self.image.blit(textsurface, (15, 15 + distance))
                distance = (i+1) * 15
            space_indicator = myfont.render('>>> Space', False, Dialog.white)
            self.image.blit(space_indicator, (150, 15 + distance))

        # add border to each edge of the image
        self.image.blit(v_border, (0,0))
        self.image.blit(v_border, (Dialog.width-border_width,0))
        self.image.blit(h_border, (0,0))
        self.image.blit(h_border, (0, Dialog.height-border_width))

    def next_frame(self):
        self.dialog_index += 1
        if self.dialog_index == self.dialog_count:
            self.is_active = False
        return self.is_active

    def update(self):
        # clear previous text
        self.image.fill(Dialog.black)
        if self.is_active == False:
            self.kill()
        if self.dialog_index >= self.dialog_count:
            self.kill()

        # print "self.dialog_index: ", self.dialog_index
        # print "self.dialog_count: ", self.dialog_count
        # print "self.is_active: ", self.is_active
        myfont = pygame.font.SysFont('Comic Sans MS', 22)

        border_width = 5
        # horizontal border
        h_border = pygame.Surface([Dialog.width, border_width])
        h_border.fill(Dialog.white)
        # vertical border
        v_border = pygame.Surface([border_width, Dialog.height])
        v_border.fill(Dialog.white)


        # render current frame text
        if self.dialog_index < self.dialog_count:
            current_text = self.text_group[self.dialog_index]
            distance = 0
            for i in range(len(current_text)):
                textsurface = myfont.render(current_text[i], False, Dialog.white)
                self.image.blit(textsurface, (15, 15 + distance))
                distance = (i+1) * 15
            space_indicator = myfont.render('>>> Space', False, Dialog.white)
            self.image.blit(space_indicator, (145, 15 + distance))

        # add border to each edge of the image
        self.image.blit(v_border, (0,0))
        self.image.blit(v_border, (Dialog.width-border_width,0))
        self.image.blit(h_border, (0,0))
        self.image.blit(h_border, (0, Dialog.height-border_width))

    def kill(self):
        pygame.sprite.Sprite.kill(self)
