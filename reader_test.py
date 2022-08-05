# test suite for pdf reader

import unittest
import os
from os.path import exists

import reader

# create instance of Scan() class to be passed to tests
scanned = reader.Scan().extract()


class ScanTest(unittest.TestCase):

    def test_read_all_pages(self):
        # print(f'Read: {scanned['read']}')
        self.assertTrue(scanned['read'] == 79)

    def test_wrote_all_pages(self):
        # print(f'Wrote: {reader.Scan().write_debug}')
        self.assertTrue(scanned['written'] == 79)

    def test_file_output(self):
        # reader.Scan().extract()
        self.assertTrue(exists(os.getcwd() + r'/out_text.txt'))


if __name__ == "__main__":
    unittest.main()
