import os
import sys
EPSILON = sys.float_info.epsilon 

class RGB:
    
    BLUE = (0, 80, 239)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    LIME = (153, 255, 0)
    ORANGE = (255, 153, 0)
    DARK_GRAY = (40, 40, 40)
    LIGHT_GRAY = (240, 240, 240)
    
    def get_rgb_list(self, steps, colors):
        minval, maxval = 1, 3
        delta = float(maxval-minval) / steps
        rgb_list= []
        for i in range(steps+1):
            val = minval + i*delta
            r, g, b = self._convert_to_rgb(minval, maxval, val, colors)
            rgb_list.append((r, g, b))
        return rgb_list

    def _convert_to_rgb(self, minval, maxval, val, colors):
        i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
        i, f = int(i_f // 1), i_f % 1
        if f < EPSILON:
            return colors[i]
        else: 
            (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
            return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

class TextPainter:
    
    rgb = RGB()
    SEQ_UNCERTAINTY = rgb.get_rgb_list(100, [rgb.DARK_GRAY, rgb.LIGHT_GRAY])
    WORD_UNCERTAINTY = rgb.get_rgb_list(5, [rgb.LIME, RGB.WHITE, rgb.WHITE, RGB.ORANGE])
    WORD_RELEVANCE = rgb.get_rgb_list(10, [rgb.BLUE, RGB.WHITE, rgb.WHITE, RGB.RED])
    
    def color_font(self, text, rgb):
        return self._colour(text, 38, rgb)
     
    def colour_background(self, text, rgb):
        return self._colour(text, 48, rgb)

    def _colour(self, text, x, rgb):
        return f"\033[{x};2;{str(rgb[0])};{str(rgb[1])};{str(rgb[2])}m{text}\033[0m"
