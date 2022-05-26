# test suite for pdf reader

import unittest

from reader import Scan


class ScanTest(unittest.TestCase):
    def test_manual_population(self):
        manual = Scan('spells.pdf')
        self.assertTrue(manual.extract() is not None)
        manual.close()


if __name__ == "__main__":
    unittest.main()
