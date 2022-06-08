# read pdf and outputs its text
# utilizes Google Tesseract OCR library

from operator import index
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


class display:  # show image and text on screen

    def __init__(self) -> None:
        pass

    # display image on screen until user presses a key
    def img(self, title, image):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setWindowTitle('image', title)
        cv2.resizeWindow('image', 1200, 900)

        # display image
        cv2.imshow('image', image)
        # maintain until user presses key
        cv2.waitKey(0)
        # destroy window on key press
        cv2.destroyAllWindows()

    # loops through captured text and arranges text line by line
    def ss_text(self, ss_details):

        # arrange text after scanning page
        parse_text = []
        word_list = []
        last_word = ''

        # loop trhough captured text on page
        for word in ss_details['text']:
            # if word captured is not empty
            if word != '':
                # add it to the line word list
                word_list.append(word)
                last_word = word
            if (last_word != '' and word == '') or (
                    word == ss_details['text'][1]):
                parse_text.append(word_list)
                word_list = []

        return parse_text


class Search:  # functions for searching text

    # initializes ss_details variable as class variable
    def __init__(self, ss_details) -> None:
        self.ss_details = ss_details

    # searching for text within image of content
    def for_text(self, search_str):

        # find all matches within one page
        results = re.findall(
            search_str, self.ss_details['text'], re.IGNORECASE)

        # in case multiple in one page
        for result in results:
            yield result

    # calculate confidence score of text from grabbed image
    def calculate_ss_confidence(self, ss_details: dict):
        # page_num  --> Page number of the detected text or item
        # block_num --> Block number of the detected text or item
        # par_num   --> Paragraph number of the detected text or item
        # line_num  --> Line number of the detected text or item
        # Convert the dict to dataFrame
        df = pd.DataFrame.from_dict(self.ss_details)
        # convert the field conf (confidence) to numberic
        df['conf'] = pd.to_numeric(df['conf'], errors='coerce')
        # Elliminate records with negative confidence
        df = df[df.conf != -1]
        # Calculate mean confidence by page
        conf = df.groupby(['page_num'])['conf'].mean().tolist()
        return conf[0]


class Serialize:  # format and eval text for storage to db
    def __init__(self) -> None:
        pass

    # appends data of PDF content line by line to pandas df
    def save_page_content(self, pdfContent, page_id, page_data):
        if page_data:
            for idx, line in enumerate(page_data, 1):
                line = ' '.join(line)
                pdfContent = pdfContent.append(
                    {'page': page_id, 'line_id': idx, 'line': line},
                    ignore_index=True
                )

        return pdfContent

    # outputs contents of padnas to csv w/ same path but diff extension (.csv)
    def save_file_content(self, pdfContent, input_file):
        content_file = os.path.join(
            os.path.dirname(input_file), os.path.splitext(
                os.path.basename(input_file))[0] + '.csv')

        pdfContent.to_csv(content_file, sep=',', index=False)

        return content_file


class Scan:  # functions for scanning the pdf files
    def __init__(self) -> None:
        pass
