from background_objects import *

class HPBar(pygame.sprite.Sprite):
	""" This class represents the HP bar. """
	_VAL_COLOR = RED
	_EMPTY_COLOR = DARK_RED
	_TEXT_COLOR = WHITE
	_BG_COLOR = BLACK
	_BUFFER_SIZE = 10
	_FONT_SIZE = 22
	_BAR_HEIGHT = HP_BAR_HEIGHT
	_BAR_LENGTH = HP_BAR_FULL_LENGTH

	def __init__(self, val_max=PLAYER_HP, pos=GAME_SCREEN_RECT.bottomleft, name='PLAYER'):
		pygame.sprite.Sprite.__init__(self, self.containers)
		self.value = val_max
		self.max_value = float(val_max)
		self.name = name

		self.pos = pos
		self.bar = pygame.Surface([self._BAR_LENGTH, self._BAR_HEIGHT])
		self.bar.fill(self._VAL_COLOR)

		pygame.font.init()
		self.font = pygame.font.SysFont('Comic Sans MS', self._FONT_SIZE)
		self.text_surface = self.get_text_surface()

		hprect = self.text_surface.get_rect()
		self.image = pygame.Surface([max(self._BAR_LENGTH, hprect.width+self._BUFFER_SIZE),
									 hprect.height+self._BUFFER_SIZE+self._BAR_HEIGHT])
		self.rect = self.image.get_rect(bottomleft=pos)

	def set_value(self, value):
		self.value = min(value, self.max_value)

	def get_text_surface(self):
		text = self.name + ' ' + str(max(0, self.value)) + '/' + str(int(self.max_value))
		return self.font.render(text, False, self._TEXT_COLOR)

	def update_bar(self):
		new_length = int(self._BAR_LENGTH * (self.value / self.max_value))
		self.bar.fill(self._VAL_COLOR, rect=Rect(0, 0, new_length, self._BAR_HEIGHT))
		self.bar.fill(self._EMPTY_COLOR, rect=Rect(new_length, 0, self._BAR_LENGTH, self._BAR_HEIGHT))

	def update(self):
		self.update_bar()
		self.text_surface = self.get_text_surface()
		self.image.fill(self._BG_COLOR)
		self.image.blit(self.text_surface, (0, 0))
		self.image.blit(self.bar, (0, self.text_surface.get_rect().height + self._BUFFER_SIZE))



class CDBar(HPBar):

	_EMPTY_COLOR = DARK_GRAY
	_FONT_SIZE = 16
	_BAR_HEIGHT = CD_BAR_HEIGHT
	_BAR_LENGTH = CD_BAR_FULL_LENGTH

	def __init__(self, val_max=PLAYER_HP, pos=GAME_SCREEN_RECT.bottomleft, name='BULLET', color=DARK_GRAY):
		HPBar.__init__(self, val_max=val_max, pos=pos, name=name)
		self._VAL_COLOR = color
		text_rect = self.text_surface.get_rect()
		self.image = pygame.Surface([CD_BAR_FULL_LENGTH + self._BUFFER_SIZE + text_rect.width,
									 max(CD_BAR_HEIGHT, text_rect.height)])
		self.rect = self.image.get_rect(bottomleft=pos)

	def get_text_surface(self):
		text = self.name
		return self.font.render(text, False, self._TEXT_COLOR)

	def update(self):
		self.update_bar()
		self.text_surface = self.get_text_surface()
		self.image.fill(self._BG_COLOR)
		self.image.blit(self.bar, (0, 0))
		self.image.blit(self.text_surface, (self._BAR_LENGTH + self._BUFFER_SIZE, 0))