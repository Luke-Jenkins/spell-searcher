# reads pdfs and outputs their text

import PyPDF2


class Scan:  # use OCR to grab array of text
    def __init__(self, pdf_title):

        self.pdf = open(pdf_title, 'rb')
        self.reader = PyPDF2.PdfReader(self.pdf)

    def extract(self):
        manual = []

        for page in self.reader.pages:
            manual.append(page)

        print(manual)

        return manual

    def close(self):
        self.pdf.close()


class Parse:  # make object for each spell
    def __init__(self) -> None:
        pass


class Serialize:  # eval & format for db
    def __init__(self) -> None:
        pass


class Store:  # insert formatted objects into db
    def __init__(self) -> None:
        pass
