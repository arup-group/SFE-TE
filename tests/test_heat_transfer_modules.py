import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import src.equivalence_methods as hrm


class TestHeatTransfer(unittest.TestCase):

    def test_calc_steel_density(self):
        self.assertEqual(hrm.SteelEC3._calc_steel_dens(), 7850)


if __name__ == '-__name__':
    unittest.main()
