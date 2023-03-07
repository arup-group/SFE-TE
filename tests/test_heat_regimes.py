import unittest
import numpy as np
import numpy.testing as npt
import src.heating_regimes as hr


class TestFlashEC1(unittest.TestCase):

    def setUp(self):
        inputs = {
            'A_c': np.array([40,60]),
            'c_ratio': np.array([0.2, 1]),
            'h_c': np.array([3,4]),
            'w_frac': np.array([0.15, 0.5]),
            'h_w_eq': np.array([1.5,3]),
            'remain_frac': np.array([0.0,0.1]),
            'q_f_d': np.array([300,300]),
            't_lim': np.array([15,20]),
            'foo': np.array([432,53]),
            'fabr_inrt' : np.array([1000,1000])}

        self.parametric = hr.FlashEC1(design_fire_inputs=inputs, crit_value=100)


    def test_perform_initial_calculations(self):
        self.parametric.perform_initial_calculations()
        expected_result = np.array([14.142, 7.746])
        npt.assert_almost_equal(self.parametric.params['c_long'], expected_result, decimal = 3)


if __name__ == '-__name__':
    unittest.main()