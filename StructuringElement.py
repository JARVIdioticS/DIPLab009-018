import numpy as np
import cmath
import sys

class StructuringElement:
    elements = None
    width = 0
    height = 0
    origin = None
    ignoreElements = None

    def __init__(self, width, height, origin):
        self.width = width
        self.height = height
        if (origin.real < 0 or origin.real >= width or origin.imag < 0 or origin.imag >= height):
            self.origin = complex(0, 0)
        else:
            self.origin = origin

        self.elements = np.zeros([width, height])
        self.ignoreElements = []