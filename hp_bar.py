from background_objects import *

class HPBar(pygame.sprite.Sprite):
	""" This class represents the HP bar. """
	_HP_BAR_FULL_LENGTH = 100
	_HP_BAR_HEIGHT = 20
	_HP_COLOR = RED
	_EMPTY_COLOR = DARK_RED
	_TEXT_COLOR = WHITE
	_BG_COLOR = BLACK
	_BUFFER_SIZE = 30

	def __init__(self, hp_max=PLAYER_HP, pos=GAME_SCREEN_RECT.bottomleft, name='PLAYER'):
		pygame.sprite.Sprite.__init__(self, self.containers)
		self.hp = hp_max
		self.max_hp = float(hp_max)
		self.name = name

		self.pos = pos
		self.bar = pygame.Surface([self._HP_BAR_FULL_LENGTH, self._HP_BAR_HEIGHT])
		self.bar.fill(self._HP_COLOR)

		pygame.font.init()
		self.font = pygame.font.SysFont('Comic Sans MS', 22)
		self.hptext = self.get_text_surface()

		hprect = self.hptext.get_rect()
		self.image = pygame.Surface([max(self._HP_BAR_FULL_LENGTH, hprect.width+self._BUFFER_SIZE),
									 hprect.height+self._BUFFER_SIZE+self._HP_BAR_HEIGHT])
		self.image.blit(self.hptext, (0, 0))
		self.image.blit(self.bar, (0, hprect.height+self._BUFFER_SIZE))
		self.rect = self.image.get_rect(bottomleft=pos)

	def set_hp(self, hp_value):
		self.hp = hp_value

	def get_text_surface(self):
		text = self.name + ' ' + str(max(0, self.hp)) + '/' + str(int(self.max_hp))
		return self.font.render(text, False, self._TEXT_COLOR)

	def update(self):
		new_length = int(self._HP_BAR_FULL_LENGTH * (self.hp / self.max_hp))
		self.bar.fill(self._HP_COLOR, rect=Rect(0, 0, new_length, self._HP_BAR_HEIGHT))
		self.bar.fill(self._EMPTY_COLOR, rect=Rect(new_length, 0, self._HP_BAR_FULL_LENGTH, self._HP_BAR_HEIGHT))
		self.hptext = self.get_text_surface()
		self.image.fill(self._BG_COLOR)
		self.image.blit(self.hptext, (0, 0))
		self.image.blit(self.bar, (0, self.hptext.get_rect().height + self._BUFFER_SIZE))
