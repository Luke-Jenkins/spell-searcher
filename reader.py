# reads pdfs and outputs their text

from distutils.log import info
from importlib.resources import path
import os
import platform
from tempfile import TemporaryDirectory
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image


class Scan:  # use OCR to grab array of text
    def __init__(self):
        # if platform.system() == "Windows":
        #     # We may need to do some additional downloading and setup...
        #     # Windows needs a PyTesseract Download
        #     # https://github.com/UB-Mannheim/tesseract/wiki/Downloading-Tesseract-OCR-Engine

        #     pytesseract.pytesseract.tesseract_cmd = (
        #         r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        #     )

        #     # Windows also needs poppler_exe
        #     self.path_to_poppler_exe = Path(r"C:\.....")

        #     # Put our output files in a sane place...
        #     out_directory = Path(r"~\Desktop").expanduser()
        # else:
        out_directory = Path(os.getcwd())
        print(f"Out Directory: {out_directory}")

        # PATH to PDF
        self.pdf_file = Path(os.getcwd() + r"/images/spells.pdf")
        print(f"PDF File Dir: {self.pdf_file}")

        # store all pages of the pdf in variable
        self.image_file_list = []

        self.text_file = out_directory / Path("out_text.txt")
        print(f"Text File Dir: {self.text_file}")

    def extract(self):
        # step 1: convert pdfs to images

        # create temp dir for images
        with TemporaryDirectory() as tempdir:

            # check platform & rad in PDF at 500 DPI
            # if platform.system() == "Windows":
            #     pdf_pages = convert_from_path(
            #         self.pdf_file, 500, poppler_path=self.path_to_poppler_exe
            #     )
            # else:
            pdfinfo = pdfinfo_from_path(
                self.pdf_file, userpw=None, poppler_path=None)

            max_pages = pdfinfo['Pages']

            # counter for file enumeration
            n = 0
            for page in range(1, max_pages + 1, 5):

                pdf_pages = convert_from_path(
                    self.pdf_file, 500, first_page=page, last_page=min(
                        page + 5 - 1, max_pages))

                # iterate through pages
                for page_enumeration, page in enumerate(pdf_pages, start=1):
                    # enumerate() "counts" the pages for us.

                    # Create a file name to store the image
                    filename = f"{tempdir}\page_{page_enumeration + n:03}.jpg"

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
                    print(f'Read page {page_enumeration + n:03}')
                n += 5

            # step 2: recognizing text from images

            # open output file in append mode
            with open(self.text_file, 'a') as output_file:
                # all contents of al images go into same file

                # iterate from 1 to toal pages
                for image_file in self.image_file_list:
                    text = pytesseract.image_to_string(Image.open(image_file))
                    output_file.write(text)
                    print(f'Wrote page {image_file[-7:-4]}')


class Parse:  # make object for each spell
    def __init__(self) -> None:
        pass


class Serialize:  # eval & format for db
    def __init__(self) -> None:
        pass


class Store:  # insert formatted objects into db
    def __init__(self) -> None:
        pass
