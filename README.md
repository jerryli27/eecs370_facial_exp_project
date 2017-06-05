# Go! Deafy!
This is the git repo for EECS 370 class project at Northwestern. It uses facial landmark recognition to build an entertaining game.

# Requirements

- Python 2.7 in Linux

- PyGame

- imutils

- dlib

- opencv

# How to Play

1. Make sure you are on Linux with the dependencies installed. 

2. Make sure there are two cameras connected to your computer, with their respective indexs. (Usually it's just 0 and 1 if you only have two cameras hooked)

3. Use the following command the start the game: 
```
python battle.py --deafy_camera_index <index of the left camera> --cat_camera_index <index of the right camera>
```
