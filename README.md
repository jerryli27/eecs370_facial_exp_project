# Go! Deafy!

This is the git repo for EECS 370 class project at Northwestern. It uses facial landmark recognition to build an entertaining 2D bullet-hell player-vs-player game. In the game, the player uses facial orientation and expression to fight against each other. Tutorials can be found at the start of the game. 

# Requirements

- Python 2.7 in Linux

- PyGame (https://www.pygame.org/news)

- imutils (https://pypi.python.org/pypi/imutils)

- dlib (https://pypi.python.org/pypi/dlib)

- opencv (http://opencv.org/)

# How to Play

0. Make sure you are on Linux and have all the dependencies installed. 

1. Download and extract
http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
at the root folder.

2. Make sure you know the indexs of your computer cameras. (Usually it's just 0 and 1 if you only have two cameras hooked)

3. Use the following command the start the game and follow the instructions. 
```
python battle.py
```
