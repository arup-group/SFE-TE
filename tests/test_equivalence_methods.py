import unittest
import os
import sys

import numpy as np
import numpy.testing as npt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
import src.equivalence_methods as eqm


class TestSteelEC3(unittest.TestCase):

    def setUp(self):
        inputs = {
            "equivalent_curve": "iso_834",
            "A_v": 100,
            "c_p": 900,
            "k_p": 0.17,
            "ro_p": 700,
            "T_amb": 20,
            "dt": 30,
            "lim_temp": 550,
            "eqv_max": 180,
            "eqv_step": 5,
            'max_itr': 10,
            'tol': 5,
            'prot_thick_range': [0.0001, 0.1, 0.0005]}
        self.eq_method = eqm.SteelEC3(**inputs)


    def test_calc_thermal_response(self):

        def exp_fxn(t, subsample_mask):
            """Exposure heat function - see test cases"""
            result = np.array([0, 0], dtype=np.float64)
            if t < 0.01:
                result[0] = 20
                result[1] = 20
            elif t <= 60:
                result[0] = 1000
                result[1] = 20 + 980*t/60
            elif t <= 120:
                result[0] = 1000
                result[1] = 1000 - 980*(t-60)/60
            else:
                result[0] = 1000
                result[1] = 20
            return result[subsample_mask]

        self.eq_method.get_equivalent_protection()

        # test t = 15 case 1.1 / 1.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=90, exposure_fxn=exp_fxn, t_final=15, sample_size=2, output_history=True, early_stop=10000)
        npt.assert_almost_equal(T_max, np.array([183.965, 36.903]), decimal=3)
        self.assertEqual(all_temps.shape, (30, 2))

        # test t = 100 case 1.1 / 1.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=90, exposure_fxn=exp_fxn, t_final=100, sample_size=2, output_history=True, early_stop=10000)
        npt.assert_almost_equal(T_max, np.array([642.064, 423.507]), decimal=3)
        self.assertEqual(all_temps.shape, (200, 2))

        # test t = 200 case 1.1 / 1.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=90, exposure_fxn=exp_fxn, t_final=200, sample_size=2, output_history=True, early_stop=10000)
        npt.assert_almost_equal(T_max, np.array([783.598, 423.506]), decimal=3)
        self.assertEqual(all_temps.shape, (400, 2))

        # test t = 200 - early stop case 1.1 / 1.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=90, exposure_fxn=exp_fxn, t_final=200, sample_size=2, output_history=True, early_stop=20)
        npt.assert_almost_equal(T_max, np.array([783.598, 423.506]), decimal=3)

        # test t = 200  case 2.1 / 2.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=120, exposure_fxn=exp_fxn, t_final=200, sample_size=2, output_history=True, early_stop=20)
        npt.assert_almost_equal(T_max, np.array([726.007, 348.803]), decimal=3)

        # test t = 200 - early stop case 2.1 / 2.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=120, exposure_fxn=exp_fxn, t_final=200, sample_size=2, output_history=True, early_stop=20)
        npt.assert_almost_equal(T_max, np.array([726.007, 348.803]), decimal=3)

        # case 3.1 / 3.2 - changes to A_v
        self.eq_method.sect_prop['A_v'] = 50
        self.eq_method.get_equivalent_protection()

        # test t = 200  case 3.1 / 3.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=90, exposure_fxn=exp_fxn, t_final=200, sample_size=2, output_history=True, early_stop=10000)
        npt.assert_almost_equal(T_max, np.array([772.039, 419.549]), decimal=3)

        # test t = 200 - early stop case 3.1 / 3.2
        T_max, all_temps = self.eq_method.calc_thermal_response(
            equiv_exp=90, exposure_fxn=exp_fxn, t_final=200, sample_size=2, output_history=True, early_stop=20)
        npt.assert_almost_equal(T_max, np.array([772.039, 419.549]), decimal=3)


    def test_calc_steel_density(self):
        self.assertEqual(self.eq_method._calc_steel_dens(), 7850)


    #TODO test_get_equivalent_protection
    #TODO test_calc_steel_hc


if __name__ == '-__name__':
    unittest.main()
