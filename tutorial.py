import pygame
from constants import *

# draw the tutorial for correct movement
def draw_correct_move_tutorial():
    _FONT_SIZE = 50
    _TEXT_COLOR = WHITE
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', _FONT_SIZE)
    text_surface = font.render("TURN YOUR FACE TO MOVE YOUR CHARACTER", False, _TEXT_COLOR)
    image = pygame.Surface([GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT])

    # load tutorial picture
    turn_left_img = pygame.image.load('data/turn_left.png')
    turn_left_img= pygame.transform.scale(turn_left_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/2))
    turn_right_img = pygame.image.load('data/turn_right.png')
    turn_right_img = pygame.transform.scale(turn_right_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/2))
    turn_up_img = pygame.image.load('data/turn_up.png')
    turn_up_img = pygame.transform.scale(turn_up_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/2))
    turn_down_img = pygame.image.load('data/turn_down.png')
    turn_down_img = pygame.transform.scale(turn_down_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/2))

    green_tick_img = pygame.image.load('data/green_tick.png')
    green_tick_img = pygame.transform.scale(green_tick_img, (80,80))

    image.blit(text_surface, (TUTORIAL_PICTURE_WIDTH-400, 50))
    image.blit(green_tick_img, (TUTORIAL_PICTURE_WIDTH-40, 100 + TUTORIAL_PICTURE_HEIGHT))

    image.blit(turn_left_img, TUTORIAL_PICTURE_LEFT_START_POS)
    image.blit(turn_right_img, TUTORIAL_PICTURE_RIGHT_START_POS)
    image.blit(turn_up_img, TUTORIAL_PICTURE_SECOND_LEFT_START_POS)
    image.blit(turn_down_img, TUTORIAL_PICTURE_SECOND_RIGHT_START_POS)

    return image

def draw_wrong_move_tutorial():
    _FONT_SIZE = 50
    _TEXT_COLOR = WHITE
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', _FONT_SIZE)
    text_surface = font.render("Move Your Body? WRONG!!!", False, _TEXT_COLOR)
    image = pygame.Surface([GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT])

    # load tutorial picture
    left_img = pygame.image.load('data/wrong_left.png')
    left_img= pygame.transform.scale(left_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT))
    right_img = pygame.image.load('data/wrong_right.png')
    right_img = pygame.transform.scale(right_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT))
    red_cross_img = pygame.image.load('data/red_wrong.png')
    red_cross_img = pygame.transform.scale(red_cross_img, (80,80))

    image.blit(text_surface, (TUTORIAL_PICTURE_WIDTH-230, 50))
    image.blit(red_cross_img, (TUTORIAL_PICTURE_WIDTH-40, 100 + TUTORIAL_PICTURE_HEIGHT))
    image.blit(left_img, TUTORIAL_PICTURE_LEFT_START_POS)
    image.blit(right_img, TUTORIAL_PICTURE_RIGHT_START_POS)

    return image

def draw_bounce_bullet_tutorial():
    _FONT_SIZE = 50
    _TEXT_COLOR = WHITE
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', _FONT_SIZE)
    text_surface = font.render("OPEN YOUR MOUTH TO FIRE BOUNCE BULLET:", False, _TEXT_COLOR)
    bottom_text = font.render("Bounce Bullet will bounce back when hit the wall", False, _TEXT_COLOR)
    image = pygame.Surface([GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT])

    # load tutorial picture
    top_img = pygame.image.load('data/close_mouth.png')
    top_img= pygame.transform.scale(top_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/3))
    mid_img = pygame.image.load('data/open_mouth.png')
    mid_img = pygame.transform.scale(mid_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/3))
    buttom_img = pygame.image.load('data/close_mouth.png')
    buttom_img = pygame.transform.scale(buttom_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/3))
    bullet_img = pygame.image.load('data/bounce_bullet.png')
    bullet_img = pygame.transform.scale(bullet_img, (TUTORIAL_PICTURE_WIDTH-200, TUTORIAL_PICTURE_HEIGHT/3))

    image.blit(text_surface, (TUTORIAL_PICTURE_WIDTH-370, 50))
    image.blit(bottom_text, (TUTORIAL_PICTURE_WIDTH-370, 150 + TUTORIAL_PICTURE_HEIGHT))

    image.blit(top_img, TUTORIAL_PICTURE_TOP_START_POS)
    image.blit(mid_img, TUTORIAL_PICTURE_MID_START_POS)
    image.blit(buttom_img, TUTORIAL_PICTURE_BUTTOM_START_POS)

    image.blit(bullet_img, TUTORIAL_BULLECT_PICTURE_START_POS)

    return image

def draw_spread_bullet_tutorial():
    _FONT_SIZE = 45
    _TEXT_COLOR = WHITE
    pygame.font.init()
    font = pygame.font.SysFont('Comic Sans MS', _FONT_SIZE)
    text_surface = font.render("STRETCH YOUR MOUTH SIDEWAYS TO FIRE SPREAD BULLET:", False, _TEXT_COLOR)
    bottom_text = font.render("Spread Bullet fires couple of bullets at once", False, _TEXT_COLOR)
    image = pygame.Surface([GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT])

    # load tutorial picture
    top_img = pygame.image.load('data/no_stretch.png')
    top_img= pygame.transform.scale(top_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/3))
    mid_img = pygame.image.load('data/stretch_mouth.png')
    mid_img = pygame.transform.scale(mid_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/3))
    buttom_img = pygame.image.load('data/no_stretch.png')
    buttom_img = pygame.transform.scale(buttom_img, (TUTORIAL_PICTURE_WIDTH, TUTORIAL_PICTURE_HEIGHT/3))
    bullet_img = pygame.image.load('data/spread_bullet.png')
    bullet_img = pygame.transform.scale(bullet_img, (TUTORIAL_PICTURE_WIDTH-200, TUTORIAL_PICTURE_HEIGHT/3))

    image.blit(text_surface, (TUTORIAL_PICTURE_WIDTH-450, 50))
    image.blit(bottom_text, (TUTORIAL_PICTURE_WIDTH-370, 150 + TUTORIAL_PICTURE_HEIGHT))

    image.blit(top_img, TUTORIAL_PICTURE_TOP_START_POS)
    image.blit(mid_img, TUTORIAL_PICTURE_MID_START_POS)
    image.blit(buttom_img, TUTORIAL_PICTURE_BUTTOM_START_POS)

    image.blit(bullet_img, TUTORIAL_BULLECT_PICTURE_START_POS)

    return image

def draw_tutorial(number):
    if number == 0:
        return draw_correct_move_tutorial()
    elif number == 1:
        return draw_wrong_move_tutorial()
    elif number == 2:
        return draw_bounce_bullet_tutorial()
    elif number == 3:
        return draw_spread_bullet_tutorial()
    else:
        print 'out of tutorials to show'
