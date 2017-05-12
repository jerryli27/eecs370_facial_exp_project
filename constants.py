from pygame.locals import *

# game constants
GAME_SCREEN_HEIGHT = 640
GAME_SCREEN_WIDTH = 640
GAME_SCREEN_RECT = Rect(0, 0, GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT)

BATTLE_SCREEN_HEIGHT=480
BATTLE_SCREEN_WIDTH=640
BATTLE_SCREEN_RECT= Rect(0, 0, BATTLE_SCREEN_WIDTH, BATTLE_SCREEN_HEIGHT)

FACIAL_LANDMARK_PREDICTOR_WIDTH = 320
CAMERA_INPUT_HEIGHT = 480
CAMERA_INPUT_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = BATTLE_SCREEN_HEIGHT / 4
CAMERA_DISPLAY_WIDTH = BATTLE_SCREEN_WIDTH / 4
CAMERA_DISPLAY_SIZE = (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)

BACKGROUND_OBJECT_HEIGHT = 32
BACKGROUND_OBJECT_WIDTH = 32

DIALOG_FRAME_COUNT = 4 # The number of total dialog frames

BULLET_SPEED = 8
BULLET_BOUNCE_H_SPEED = 3

INITIAL_GRAVITY = 2  # pixel/second^2
MAX_JUMP_CHARGE = 1  # The number of time the object can jump
MAX_JUMP_SPEED = 30
INITIAL_DX = 0
INITIAL_DY = 0
STD_DX = -1
# The gravity factor works similarly to BLINK_JUMP_SPEED_FACTOR. The gravity is decreased by this factor when feature score
# exceeds the lower threshold, so that when the mouth opens larger, Deafy falls slower.
GRAVITY_FACTOR = 1
MIN_GRAVITY = 2.0 # 0.5


# UI object specifics
MAX_FPS = 30
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

# Facial landmark constants
MOUTH_SCORE_SHOOT_THRESHOLD = 1.5  # The mouth score has to be at least this big for Deafy to start shooting.
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

assert BATTLE_SCREEN_HEIGHT % BACKGROUND_OBJECT_HEIGHT == 0 and BATTLE_SCREEN_WIDTH % BACKGROUND_OBJECT_WIDTH == 0
