# reads pdfs and outputs their text

from distutils.log import info
from importlib.resources import path
import os
from os.path import exists
from tqdm import tqdm 
import platform
from tempfile import TemporaryDirectory
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image


class Scan:  # use OCR to grab array of text
    def __init__(self):

        out_directory = Path(os.getcwd())
        # print(f"Out Directory: {out_directory}")

        # PATH to PDF
        self.pdf_file = Path(os.getcwd() + r"/images/spells.pdf")
        # print(f"PDF File Dir: {self.pdf_file}")

        # store all pages of the pdf in variable
        self.image_file_list = []

        self.text_file = out_directory / Path("out_text.txt")
        print(f"Text File Dir: {self.text_file}")

    def extract(self):
        # debug check
        # if exists(os.getcwd() + r'/out_text.txt') is True:
        #     return

        # step 1: convert pdfs to images

        # create temp dir for images
        with TemporaryDirectory() as tempdir:
            pdfinfo = pdfinfo_from_path(
                self.pdf_file, userpw=None, poppler_path=None)

            max_pages = pdfinfo['Pages']

            # counter for file enumeration
            n = 0

            for page in tqdm(range(1, max_pages + 1, 5), desc='Reading'):

                pdf_pages = convert_from_path(
                    self.pdf_file, 500, first_page=page, last_page=min(
                        page + 5 - 1, max_pages))

                # iterate through pages
                for page_enumeration, page in enumerate(pdf_pages, start=1):
                    # enumerate() "counts" the pages for us.

                    # Create a file name to store the image
                    filename = f"{tempdir}/page_{page_enumeration + n:03}.jpg"

                    # Declaring filename for each page of PDF as JPG
                    # For each page, filename will be:
                    # PDF page 1 -> page_001.jpg
                    # PDF page 2 -> page_002.jpg
                    # PDF page 3 -> page_003.jpg
                    # ....
                    # PDF page n -> page_00n.jpg

                    # Save the image of the page in system
                    page.save(filename, "JPEG")
                    self.image_file_list.append(filename)
                    self.read_debug = page_enumeration + n
                    # return self.read_debug
                n += 5

            # step 2: recognizing text from images

            # open output file in append mode
            with open(self.text_file, 'a') as output_file:
                # all contents of al images go into same file

                # iterate from 1 to toal pages
                for image_file in tqdm(self.image_file_list, desc='Writing'):
                    text = pytesseract.image_to_string(Image.open(image_file))
                    output_file.write(text)
                    self.write_debug = int(image_file[-7:-4])
                    # return self.write_debug

        # return pages read and written
        scanned = {'read': self.read_debug, 'written': self.write_debug}
        return scanned


class Spell:  # make object for each spell
    def __init__(self):
        # spell structures
        self.title = ''
        self.level = ''
        self.cast_time = ''
        self.range = ''
        self.components = ''
        self.duration = ''
        self.description = ''

        # source text file
        self.text_file = Path(os.getcwd() + r'/out_text.txt')
        # array to hold text
        self.spell_text = []

    def preprocess(self):  # remove lines & store text in array

        with open(self.text_file) as text_file:

            for line in text_file:
                # ignore blank lines
                if line != '\n':
                    # store line in array
                    self.spell_text.append(line)

        text_file.close()
    
    def parse(self):  # read through source text for spell parts
        
        for line in self.spell_text:
            if line.isupper() = True:
                self.title = line[0:len(line)-1]

class Serialize:  # eval & format for db
    def __init__(self) -> None:
        pass


class Store:  # insert formatted objects into db
    def __init__(self) -> None:
        pass
