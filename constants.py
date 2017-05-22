from pygame.locals import *
import pygame

# Debug constant. Used to print debugging messages if needed. Can be set by using the debug_level flag. See that flag
# for detailed description.
DEBUG_LEVEL = 0
DEBUG_PRINT_ONLY_CRUCIAL = 0
DEBUG_PRINT_UNEXPECTED_ERROR = 1
DEBUG_PRINT_ALL = 2

# game constants
GAME_SCREEN_HEIGHT = 640
GAME_SCREEN_WIDTH = 640
GAME_SCREEN_RECT = Rect(0, 0, GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT)

BATTLE_SCREEN_HEIGHT=480
BATTLE_SCREEN_WIDTH=640
BATTLE_SCREEN_RECT= Rect(0, 0, BATTLE_SCREEN_WIDTH, BATTLE_SCREEN_HEIGHT)
BATTLE_SCREEN_SIZE= (BATTLE_SCREEN_WIDTH, BATTLE_SCREEN_HEIGHT)

HP_BAR_DEAFY_BOTTOMLEFT = (60, 600)
HP_BAR_CAT_BOTTOMLEFT = (480, 600)

FACE_DEAFY_BOTTOMLEFT = (200, 635)
FACE_CAT_BOTTOMLEFT = (340, 635)

FACIAL_LANDMARK_PREDICTOR_WIDTH = 320
# Command to find supported camera resolution: v4l2-ctl --list-formats-ext
CAMERA_INPUT_HEIGHT = 480 / 2 # 720 # 480 * 2
CAMERA_INPUT_WIDTH = 640 / 2 # 1280 # 640 * 2
CAMERA_INPUT_SIZE = (CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT)
CAMERA_DISPLAY_HEIGHT = BATTLE_SCREEN_HEIGHT / 4
CAMERA_DISPLAY_WIDTH = BATTLE_SCREEN_WIDTH / 4
CAMERA_DISPLAY_SIZE = (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)

DEAFY_CAMERA_DISPLAY_LOCATION = (0, BATTLE_SCREEN_HEIGHT - CAMERA_DISPLAY_HEIGHT)
CAT_CAMERA_DISPLAY_LOCATION = (BATTLE_SCREEN_WIDTH - CAMERA_DISPLAY_WIDTH, BATTLE_SCREEN_HEIGHT - CAMERA_DISPLAY_HEIGHT)


BACKGROUND_OBJECT_HEIGHT = 32
BACKGROUND_OBJECT_WIDTH = 32

PHOTO_DISPLAY_WIDTH = BATTLE_SCREEN_WIDTH / 10
PHOTO_DISPLAY_HEIGHT = (GAME_SCREEN_HEIGHT - BATTLE_SCREEN_HEIGHT) / 4
PHOTO_DISPLAY_SIZE = (PHOTO_DISPLAY_WIDTH, PHOTO_DISPLAY_HEIGHT)
PHOTO_DISPLAY_DEAFY_BOTTOMLEFT = (0,
                                  BATTLE_SCREEN_HEIGHT + (GAME_SCREEN_HEIGHT - BATTLE_SCREEN_HEIGHT) * 3 / 4)
PHOTO_DISPLAY_DEAFY_DELTA = (PHOTO_DISPLAY_WIDTH, 0)
PHOTO_DISPLAY_CAT_BOTTOMLEFT = (BATTLE_SCREEN_WIDTH - BATTLE_SCREEN_WIDTH / 10,
                                BATTLE_SCREEN_HEIGHT + (GAME_SCREEN_HEIGHT - BATTLE_SCREEN_HEIGHT) * 3 / 4)
PHOTO_DISPLAY_CAT_DELTA = (-PHOTO_DISPLAY_WIDTH, 0)

DIALOG_FRAME_COUNT = 6 # The number of total dialog frames

BULLET_SPEED = 8
BULLET_BOUNCE_H_SPEED = 3

PLAYER_AP = 30
PLAYER_HP = 100

INITIAL_GRAVITY = 2  # pixel/second^2
MAX_JUMP_CHARGE = 1  # The number of time the object can jump
MAX_JUMP_SPEED = 30
INITIAL_DX = 0
INITIAL_DY = 0
STD_DX = -1
DEFAULT_SPEED = 5  # Default speed for both cat and deafy in pixel per second.
# The gravity factor works similarly to BLINK_JUMP_SPEED_FACTOR. The gravity is decreased by this factor when feature score
# exceeds the lower threshold, so that when the mouth opens larger, Deafy falls slower.
GRAVITY_FACTOR = 1
MIN_GRAVITY = 2.0 # 0.5

# Direction key constants.
WSAD_KEY_CONSTANTS = [K_w, K_s, K_a, K_d]
ARROW_KEY_CONSTANTS = [K_UP, K_DOWN, K_LEFT, K_RIGHT]

# UI object specifics
# FPS should be at least 10 for good game experience, ideally 20-30.
MAX_FPS = 30
FPS_BLIT_BOTTOM_LEFT = (0 + 5, 25) # (0 + 5, GAME_SCREEN_HEIGHT - 5)
# The y goes from top to bottom starting at 0.
GROUND_LEVEL = BATTLE_SCREEN_HEIGHT * 11 / 15
GROUND_Y_LIMITS = (GROUND_LEVEL, BATTLE_SCREEN_HEIGHT)
SKY_Y_LIMITS = (0,GROUND_LEVEL)
DEAFY_SCREEN_POS = (BATTLE_SCREEN_WIDTH / 8, GROUND_LEVEL)
CAT_SCREEN_POS = (BATTLE_SCREEN_WIDTH * 7 / 8, GROUND_LEVEL)
BULLET_SIZE = 4

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARK_RED = (100, 0, 0)
DARK_BLUE = (0, 0, 100)
LIGHT_RED = (255, 200, 200)
LIGHT_BLUE = (200, 200, 255)

# Facial landmark constants
SELLION_INDEX = 27
MOUTH_SCORE_SHOOT_THRESHOLD = 0.8  # The mouth score has to be at least this big for Deafy to start shooting.
MOUTH_SCORE_RECHARGE_THRESHOLD = 0.4  # The mouth score has to be at least this big for Deafy to start shooting.
# The jump speed is the facial feature score times this factor: score * BLINK_JUMP_SPEED_FACTOR pixels per second.
BLINK_JUMP_SPEED_FACTOR = 3
# This controls the speed at which Deafy moves. (Temporary solution for the demo)
MOUTH_SCORE_SPEED_THRESHOLDS = [(0.3, -2), (0.4, -4), (0.6, -6), (0.8, -8)]
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# The lower and upper threshold for pose-controlled moving.
POSE_MOVE_LOWER_THRESHOLD = 10
POSE_MOVE_UPPER_THRESHOLD = 100

assert BATTLE_SCREEN_HEIGHT % BACKGROUND_OBJECT_HEIGHT == 0 and BATTLE_SCREEN_WIDTH % BACKGROUND_OBJECT_WIDTH == 0

# Music
# You can have several User Events, so make a separate Id for each one
BGM_CHANNEL_ID = 0
END_BGM_EVENT = pygame.USEREVENT + 0    # ID for music Event