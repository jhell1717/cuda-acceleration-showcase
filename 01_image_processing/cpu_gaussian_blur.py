import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import time

def load_image(path,grayscale=True):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def cpu_gaussian_blur(img,sigma=3):
    blurred = gaussian_filter(img,sigma=sigma)
    return blurred
    