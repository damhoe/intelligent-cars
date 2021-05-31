"""
Graphics class.

@author: Damian Hoedtke
@date: May, 2021

1 inch = 2.54 cm

"""
import numpy as np

class Graphics:

    def __init__(self, dpi, scale):
        self.dpi = dpi
        self.scale = scale

    def cm2px(self, number):
        return int(number * self.dpi * 1.0 / 2.54)

    def cm2px_array(self, array):
        a = array * self.dpi * 1.0 / 2.54
        return a.astype(np.int32)

    def get_scale(self):
        return self.scale * self.dpi * 1.0 / 2.54

    def convert(self, a, key='m'):

        if key == 'm':
            factor = 1.0 * 1e2
        elif key == 'cm':
            factor = 1.0
        else:
            raise Exception("Unknown key encounter in 'Graphics.scaled()'.")


        # scale to cm
        b = a * self.scale * factor

        if type(a) == np.ndarray:
            return self.cm2px_array(b)

        return self.cm2px(b)
