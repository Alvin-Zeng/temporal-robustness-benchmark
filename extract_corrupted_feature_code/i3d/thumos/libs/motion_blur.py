# to install 'wand', use:  sudo apt-get install libmagickwand-dev

from wand.api import library as wandlibrary
from wand.image import Image as WandImage
import cv2
import numpy as np

# Extend wand.image.Image class to include method signature

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def motionblur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    _, image_buffer = cv2.imencode('.jpg', x)
    x = MotionImage(blob=image_buffer.tobytes())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_COLOR)
    return np.clip(x, 0, 255)

def motion_blur(image):
    scale = 255. / (image + 1).max()
    image = (image + 1) * scale
    image = motionblur(image, 5) / scale - 1
    return image