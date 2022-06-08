# simple ocr reader powered by Tesseract

import cv2
import os
import argparse
import pytesseract
from PIL import Image

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
                required=True,
                help='Path to the image folder')
ap.add_argument('-p', '--pre_processor',
                default='thresh',
                help='the preprocessor usage')
args = vars(ap.parse_args())

# read the image with text
images = cv2.imread(args['image'])

# convert to grayscale
gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

# checking whether thresh or blur
if args['pre_preprocessor'] == 'thresh':
    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
if args['pre_preprocessor'] == 'blur':
    cv2.medianBlur(gray, 3)

# memory usage with image i.e. adding image to memory
filename = '{}.jpg'.format(os.getpid())
cv2.imwrite(filename, gray)
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output image
cv2.imshow("Image Input", images)
cv2.imshow("Output In Grayscale", gray)
cv2.waitKey(0)
