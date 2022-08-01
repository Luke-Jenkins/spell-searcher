# test suite for pdf reader

from os import getcwd
import unittest
import os
from os.path import exists

import reader


class ScanTest(unittest.TestCase, reader.Scan):

    def test_file_output(self):
        reader.Scan().extract()
        self.assertTrue(exists(os.getcwd() + r'/out_text.txt'))

    def test_read_all_pages(self):
        print(f'Read: {reader.Scan().read_debug}')
        self.assertTrue(reader.Scan().read_debug == 79)

    def test_wrote_all_pages(self):
        print(f'Wrote: {reader.Scan().write_debug}')
        self.assertTrue(reader.Scan().write_debug == 79)


if __name__ == "__main__":
    unittest.main()
