import unittest
import src.heat_transfer_models as hrm


class TestHeatTransfer(unittest.TestCase):

    def test_calc_steel_density(self):
        self.assertEqual(hrm.SteelEC3._calc_steel_dens(), 7850)


if __name__ == '-__name__':
    unittest.main()
