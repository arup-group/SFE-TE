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
            'fabr_inrt' : np.array([1000, 1000])}
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

        #Test max_temp
        expected_result= np.array([859.632, 665.721])
        # print(self.parametric.params['max_temp_t'])
        npt.assert_almost_equal(self.parametric.params['max_temp'], expected_result, decimal=0) # does not work with decimals=3


    # test temperature for heating phase - revisit
    def test_calc_heat_phase_temp(self):
        self.parametric.perform_initial_calculations()


        expected_heat_phase_temp = {3: [626.0, 232.0],
                                    5: [716.0, 336.0]}

        for key in expected_heat_phase_temp:
            self.parametric.perform_initial_calculations()

            expected_result = expected_heat_phase_temp.get(key)

            t = key
            _, t_str_heat, _ = self.parametric.get_exposure(t)

            npt.assert_almost_equal(self.parametric._calc_heat_phase_temp(t_str_heat), expected_result, err_msg=f"error for t = {key}", decimal=0)

    # Test for get exposure
    def test_get_exposure(self):
        self.parametric.perform_initial_calculations()

        # test cool phase temp
        expected_cool_phase_temp = {20: [756.0, 666.0],
                                    50: [80.0, 20.0]}

        for key in expected_cool_phase_temp:

            self.parametric.perform_initial_calculations()

            expected_result = expected_cool_phase_temp.get(key)

            t = key
            _, _, cool_phase_temp = self.parametric.get_exposure(t)

            npt.assert_almost_equal(cool_phase_temp, expected_result, err_msg=f"error at t = {t}", decimal=0)


class TestTravelingISO16733(unittest.TestCase):
    def setUp(self):
        inputs = {
            'A_c': np.array([100, 200]),
            'h_c': np.array([3, 4]),
            'c_ratio': np.array([0.2, 1]),
            'q_f_d': np.array([300, 300]),
            'Q': np.array([]),
            'spr_rate': np.array([]),
            'flap_angle': np.array([ ]),
            'T_nf_max': np.array([])}
        crit_value = 100

        self.traveling = hr.TravelingISO16733(design_fire_inputs=inputs, crit_value=crit_value)

    def test_perform_initial_calculations(self):
        self.traveling.perform_initial_calculations()

        # Test c_long
        expected_result = np.array([])
        npt.assert_almost_equal(self.traveling.params['c_long'], expected_result, decimal=3)

        # Test c_short
        expected_result = np.array([])
        npt.assert_almost_equal(self.traveling.params['c_short'], expected_result, decimal=3)

        # Test t_b (fire burning time)
        expected_result = np.array([])
        npt.assert_almost_equal(self.traveling.params['t_b'], expected_result, decimal=3)

        # Test L_f (
        expected_result = np.array([])
        npt.assert_almost_equal(self.traveling.params['L_f'], expected_result, decimal=3)

        # Test burnout
        expected_result = np.array([])
        npt.assert_almost_equal(self.traveling.params['burnout'], expected_result, decimal=3)

        # Test f
        expected_result = np.array([])
        npt.assert_almost_equal(self.traveling.params['f'], expected_result, decimal=3)




if __name__ == '__main__':
    unittest.main()