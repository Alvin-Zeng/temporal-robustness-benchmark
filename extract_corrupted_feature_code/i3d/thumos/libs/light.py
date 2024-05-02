import cv2
import numpy as np

def add_light(image):
    scale = 255 / (image + 1).max()
    image = scale * (image + 1)
    exposure_intensity = 100.0
    overexposed_image = cv2.add(image, np.array([exposure_intensity]))
    overexposed_image = np.where(overexposed_image > 255, 255, overexposed_image)
    return overexposed_image / scale - 1