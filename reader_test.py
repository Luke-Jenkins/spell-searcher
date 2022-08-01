# test suite for pdf reader

from os import getcwd
import unittest
import os
from os.path import exists

import reader


class ScanTest(unittest.TestCase):
    def test_manual_population(self):
        reader.Scan().extract()
        self.assertTrue(exists(os.getcwd() + r'/out_text.txt'))


if __name__ == "__main__":
    unittest.main()
