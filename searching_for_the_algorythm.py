import cv2
import numpy as np

class Person:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_postition(self, x, y):
        self.x = x
        self.y = y


