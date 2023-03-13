import unittest
import numpy as np
import numpy.testing as npt
import src.heating_regimes as hr


class TestFlashEC1(unittest.TestCase):

    def setUp(self):
        inputs = {
            'A_c': np.array([40, 60]),
            'c_ratio': np.array([0.2, 1]),
            'h_c': np.array([3, 4]),
            'w_frac': np.array([0.15, 0.5]),
            'h_w_eq': np.array([1.5, 3]),
            'remain_frac': np.array([0.0, 0.1]),
            'q_f_d': np.array([300, 300]),
            't_lim': np.array([15, 20]),
            # 'foo': np.array([432, 53]),  #unclear
            'fabr_inrt' : np.array([1000, 1000])}    #unclear - thermal absorptivity?
        crit_value = 100

        self.parametric = hr.FlashEC1(design_fire_inputs=inputs, crit_value=crit_value)

    def test_perform_initial_calculations(self):
        self.parametric.perform_initial_calculations()

        # Test c_long calculation
        expected_result = np.array([14.142, 7.746])
        npt.assert_almost_equal(self.parametric.params['c_long'], expected_result, decimal = 3)

        # Test c_short calculation
        expected_result = np.array([2.828 , 7.746])
        npt.assert_almost_equal(self.parametric.params['c_short'], expected_result, decimal = 3)

        #Test c_perim
        expected_result = np.array([33.941, 30.984])
        npt.assert_almost_equal(self.parametric.params['c_perim'], expected_result, decimal = 3)

        #Test A_t
        expected_result = np.array([181.823, 243.935])
        npt.assert_almost_equal(self.parametric.params['A_t'], expected_result, decimal = 3)

        #Test A_v
        expected_result = np.array([7.637, 46.476])
        npt.assert_almost_equal(self.parametric.params['A_v'], expected_result, decimal=3)

        #Test Of_max
        expected_result = np.array([0.051, 0.200])
        npt.assert_almost_equal(self.parametric.params['Of_max'], expected_result, decimal=3)

        #Test Of
        expected_result = np.array([0.051, 0.180])
        npt.assert_almost_equal(self.parametric.params['Of'], expected_result, decimal=3)

        #Test q_t_d
        expected_result = np.array([65.998, 73.790])
        npt.assert_almost_equal(self.parametric.params['q_t_d'], expected_result, decimal=3)

        #Test Of_lim
        expected_result = np.array([0.026, 0.022])
        npt.assert_almost_equal(self.parametric.params['Of_lim'], expected_result, decimal=3)

        #Test GA
        expected_result = np.array([2.225, 27.248])
        npt.assert_almost_equal(self.parametric.params['GA'], expected_result, decimal=3)

        #Test k
        expected_result = np.array([0.995, 0.991])
        npt.assert_almost_equal(self.parametric.params['k'], expected_result, decimal=3)

        #Test GA_lim
        expected_result = np.array([0.583, 0.408])
        npt.assert_almost_equal(self.parametric.params['GA_lim'], expected_result, decimal=3)

        #Test t_max_vent
        expected_result = np.array([0.257, 0.082])
        npt.assert_almost_equal(self.parametric.params['t_max_vent'], expected_result, decimal=3)

        #Test regime
        expected_result = np.array(['V', 'F'])
        npt.assert_equal(self.parametric.params['regime'], expected_result)

        #Test max_temp_t
        expected_result = np.array([0.257, 0.333])
        npt.assert_almost_equal(self.parametric.params['max_temp_t'], expected_result, decimal=3)

        #Test t_star_max
        expected_result = np.array([0.571, 0.136])
        npt.assert_almost_equal(self.parametric.params['t_str_max_heat'], expected_result, decimal=3)


    # def test_get_exposure(self):
    #     self.parametric.perform_initial_calculations()
    #
    #     #Test at t = 5
    #     t = 5
    #     exposure, t_str_heat = self.parametric.get_exposure(t)
    #     expected_result = np.array([14.142, 7.746])
    #     npt.assert_almost_equal(exposure, expected_result, decimal = 3)
    #
    #     #Test at t = 50
    #     t = 50
    #     exposure, t_str_heat = self.parametric.get_exposure(t)
    #     expected_result = np.array([14.142, 7.746])
    #     npt.assert_almost_equal(exposure, expected_result, decimal = 3)
    #
    # def test_max_temp(self):
    #     self.parametric.perform_initial_calculations()
    #
    #     t = 5
    #     exposure, t_str_heat = self.parametric.get_exposure(t)
    #     self.parametric.
    #     expected_result = np.array([14.142, 7.746])
    #     npt.assert_almost_equal(, expected_result, decimal = 3)
    #
    #
    #     #Test at
    # def test_heat_phase_temp(self):


if __name__ == '__main__':
    unittest.main()