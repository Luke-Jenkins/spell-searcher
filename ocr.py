# read pdf and outputs its text
# utilizes Google Tesseract OCR library

import os
import re
import argparse
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import fitz
from io import BytesIO
from PIL import Image
import pandas as pd
import filetype

# path of Tesseact OCR engine
tesseract_path = '/usr/local/Cellar/tesseract/5.1.0/bin/tesseract'

# include tesseract executable
pytesseract.pytesseract.tesseract_cmd = tesseract_path


class Preprocess:  # make image more readable for OCR
    def __init__(self):
        self.kernel = np.ones((5, 5), np.unit8)

    # convert pixmap buffer into numpy array
    def pix2np(self, pix):
        image = np.frombuffer(pix.samples, dtype=np.unit8).reshape(
            pix.h, pix.w, pix.n)

        try:
            image = np.ascontiguousarray(image[..., [2, 1, 0]])
        except IndexError:
            # convert Gray to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = np.ascontiguousarray(image[..., [2, 1, 0]])

        return image

    # convert to grayscale
    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove noise
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def threshold(self, image):
        return cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # dilation
    def dilate(self, image):
        return cv2.dilate(image, self.kernel, iterations=1)

    # erosion
    def erode(self, image):
        return cv2.erode(image, self.kernel, iterations=1)

    # opening -- erosion followed by dilation
    def opening(self, image):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE)

        return rotated

    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # pre-processes the image and creates binary output
    def convert_img2bin(self, image):
        # convert image to grayscale
        output_img = self.grayscale(image)

        # invert grayscale image by flipping pixel values
        output_img = cv2.bitwise_not(output_img)

        # converting image to binary by Thresholding
        # shows clear separation between wht and blk pixels
        output_img = self.threshold(output_img)

        return output_img


# display image on screen until user presses a key
def display_img(title, image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('image', title)
    cv2.resizeWindow('image', 1200, 900)

    # display image
    cv2.imshow('image', image)
    # maintain until user presses key
    cv2.waitKey(0)
    # destroy window on key press
    cv2.destroyAllWindows()
