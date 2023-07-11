import unittest
import numpy as np
import numpy.testing as npt
import src.heating_regimes as hr


class TestUniEC1(unittest.TestCase):

    def setUp(self):
        inputs = {
            'A_c': np.array([40, 60, 1000]),
            'c_ratio': np.array([0.2, 1, 0]),
            'h_c': np.array([3, 4, 0]),
            'w_frac': np.array([0.15, 0.5, 0]),
            'remain_frac': np.array([0.0, 0.1, 0]),
            'q_f_d': np.array([300, 300, 0]),
            't_lim': np.array([15, 20, 0]),
            'w_frac_h': np.array([0.5, 0.75, 0]),
            'fabr_inrt' : np.array([1000, 1000, 0]),
            'foo': np.array([1000, 1000, 0])}
        crit_value = 100
        Of_limits = [0.02, 0.2]
        max_fire_duration = 1000

        self.parametric = hr.UniEC1(
            design_fire_inputs=inputs,
            crit_value=crit_value,
            max_fire_duration=max_fire_duration,
            Of_limits=Of_limits)

    def test_get_relevant_design_fire_indeces(self):
        npt.assert_equal(self.parametric.relevent_df_indices, np.array([0, 1]))
        self.assertFalse(self.parametric.is_empty, 'Issue with empty method definition')

    def test_get_parameters(self):
        self.assertFalse('foo' in self.parametric.params)
        self.assertEqual(len(self.parametric.params), 9)

    def test_perform_initial_calculations(self):

        # Test _calc_comp_sides calculations
        self.parametric._calc_comp_sides()
        npt.assert_almost_equal(self.parametric.params['c_long'], np.array([14.142, 7.746]), decimal=3)
        npt.assert_almost_equal(self.parametric.params['c_short'], np.array([2.828, 7.746]), decimal=3)

        # Test c_perim
        self.parametric._calc_perimeter()
        npt.assert_almost_equal(self.parametric.params['c_perim'], np.array([33.941, 30.984]), decimal=3)

        # Test A_t - total enclosure area
        self.parametric._calc_total_area_enclosure()
        npt.assert_almost_equal(self.parametric.params['A_t'], np.array([181.823, 243.935]), decimal=3)

        # Test h_w_eq - average window height
        self.parametric._calc_average_window_height()
        npt.assert_almost_equal(self.parametric.params['h_w_eq'], np.array([1.500, 3.000]), decimal=3)

        # Test A_v - total ventilation area
        self.parametric._calc_total_ventilation_area()
        npt.assert_almost_equal(self.parametric.params['A_v'], np.array([7.637, 46.476]), decimal=3)

        # Test Of - breakage opening factor
        self.parametric._calc_max_open_factor()
        self.parametric._calc_open_factor_breakage()
        npt.assert_almost_equal(self.parametric.params['Of'], np.array([0.051, 0.297]), decimal=3)

        # Test apply open factor limits
        self.parametric._apply_open_factor_limits()
        npt.assert_almost_equal(self.parametric.params['Of'], np.array([0.051, self.parametric.Of_limits[1]]), decimal=3)

        # Test GA - test calc Ga
        self.parametric._calc_GA_factor()
        npt.assert_almost_equal(self.parametric.params['GA'], np.array([2.225, 33.640]), decimal=3)

        # Test q_t_d - total surface area fuel density
        self.parametric._calc_total_surface_area_fuel_density()
        npt.assert_almost_equal(self.parametric.params['q_t_d'], np.array([65.998, 73.790]), decimal=3)

        # Test t_max_vent - time to max temp for vent controlled fire
        self.parametric._calc_t_max_vent()
        npt.assert_almost_equal(self.parametric.params['t_max_vent'], np.array([0.257, 0.074]), decimal=3)

        # Test t_max_fuel - time to max temp for fuel controlled fire
        self.parametric._calc_t_max_fuel()
        npt.assert_almost_equal(self.parametric.params['t_max_fuel'], np.array([0.250, 0.333]), decimal=3)

        # Test Of_lim - Open factor for fuel controlled
        self.parametric._calc_open_factor_fuel()
        npt.assert_almost_equal(self.parametric.params['Of_lim'],  np.array([0.026, 0.022]), decimal=3)

        # Test k and Test GA_lim - K factor for GA_lim - NOTE: Ga_lim is modified with k
        self.parametric._calc_GA_lim_factor()
        self.parametric._calc_GA_lim_k_mod()
        npt.assert_almost_equal(self.parametric.params['k'], np.array([0.995, 0.991]), decimal=3)
        npt.assert_almost_equal(self.parametric.params['GA_lim'], np.array([0.583, 0.408]), decimal=3)

        # Test burning regime - checking for fuel or vent control
        self.parametric._define_burning_regime()
        npt.assert_equal(self.parametric.params['regime'], np.array(['V', 'F']))

        # Test max_temp_t - time of maximum temperature 
        self.parametric._calc_max_temp_time()
        npt.assert_almost_equal(self.parametric.params['max_temp_t'], np.array([0.257, 0.333]), decimal=3)

        # Test t_star_max
        self.parametric._calc_t_star_max()
        npt.assert_almost_equal(self.parametric.params['t_str_max_heat'], np.array([0.571, 0.136]), decimal=3)

        # Test max_temp - maximum temperature
        self.parametric._calc_max_temp()
        npt.assert_almost_equal(self.parametric.params['max_temp'], np.array([859.632, 665.721]), decimal=3)

        # Test fire duration - fire duration TODO: Add expected values for fire duration
        self.parametric._calc_fire_duration()
        npt.assert_almost_equal(self.parametric.params['burnout'], np.array([52.676, 24.607]), decimal=3)

        # Test _define_max_gas_temp - for external class use
        self.parametric._define_max_gas_temp()
        npt.assert_almost_equal(self.parametric.params['max_gas_temp'], np.array([859.632, 665.721]), decimal=3)



    # test temperature for heating phase - revisit
    # def test_calc_heat_phase_temp(self):
    #     self.parametric.perform_initial_calculations()
    #
    #
    #     expected_heat_phase_temp = {3: [626.0, 232.0],
    #                                 5: [716.0, 336.0]}
    #
    #     for key in expected_heat_phase_temp:
    #         self.parametric.perform_initial_calculations()
    #
    #         expected_result = expected_heat_phase_temp.get(key)
    #
    #         t = key
    #         _, t_str_heat, _ = self.parametric.get_exposure(t)
    #
    #         npt.assert_almost_equal(self.parametric._calc_heat_phase_temp(t_str_heat), expected_result, err_msg=f"error for t = {key}", decimal=0)
    #
    # # Test for get exposure
    # def test_get_exposure(self):
    #     self.parametric.perform_initial_calculations()
    #
    #     # test cool phase temp
    #     expected_cool_phase_temp = {20: [756.0, 666.0],
    #                                 50: [80.0, 20.0]}
    #
    #     for key in expected_cool_phase_temp:
    #
    #         self.parametric.perform_initial_calculations()
    #
    #         expected_result = expected_cool_phase_temp.get(key)
    #
    #         t = key
    #         _, _, cool_phase_temp = self.parametric.get_exposure(t)
    #
    #         npt.assert_almost_equal(cool_phase_temp, expected_result, err_msg=f"error at t = {t}", decimal=0)

    #TODO check bad samples - to be updated once such tests are created


class TestTravelingISO16733(unittest.TestCase):
    def setUp(self):
        inputs = {
            'A_c': np.array([500,500]),
            'h_c': np.array([4, 4]),
            'c_ratio': np.array([0.8, 0.8]),
            'q_f_d': np.array([570, 570]),
            'Q': np.array([290, 290]),
            'spr_rate': np.array([2.544, 2.544]),
            'flap_angle': np.array([6.5, 0.0]),
            'T_nf_max': np.array([1200, 1200]),
            'T_amb': np.array([20, 20])}
        crit_value = 100
        assess_loc = 0.8
        max_travel_path = 25
        max_fire_duration = 1000

        self.traveling = hr.TravelingISO16733(
            design_fire_inputs = inputs,
            crit_value = crit_value,
            assess_loc = assess_loc,
            max_travel_path = max_travel_path,
            max_fire_duration = max_fire_duration)

    def test_perform_initial_calculations(self):
        #self.traveling.perform_initial_calculations() NOTE: commented out to test functions individually


        # TODO rewrite this that each method of intial calculation is tested
        # Add test cases, add a test case with a flapping angle of zero

        # Test c_long and c_short
        self.traveling._calc_comp_sides()
        npt.assert_almost_equal(self.traveling.params['c_long'], np.array([25.000, 25.000]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['c_short'], np.array([20.000, 20.000]), decimal=3)
        
        # Test x_loc - position along fire path
        self.traveling._calc_assess_loc()
        npt.assert_almost_equal(self.traveling.params['x_loc'], np.array([20.000, 20.000]), decimal=3)

        # Test t_b - fire burning time 
        self.traveling._calc_burning_time()
        npt.assert_almost_equal(self.traveling.params['t_b'], np.array([1965.517, 1965.517]), decimal=3)

        # Test L_f - fire base length 
        self.traveling._calc_fire_base_length()
        npt.assert_almost_equal(self.traveling.params['L_f'], np.array([5.000, 5.000]), decimal=3)

        # Test A_f - fire base area 
        self.traveling._calc_fire_base_area()
        npt.assert_almost_equal(self.traveling.params['A_f'], np.array([100.006, 100.006]), decimal=3)
        
        # Test L_str - Relative fire size 
        self.traveling._calc_relative_fire_size()
        npt.assert_almost_equal(self.traveling.params['L_str'], np.array([0.200, 0.200]), decimal=3)

        # Test burnout - burning time for whole travelling fire path 
        self.traveling._calc_burnout()
        npt.assert_almost_equal(self.traveling.params['burnout'], np.array([196.543, 196.543]), decimal=3)

        # Test f - flapping length
        self.traveling._calc_flap_l()
        npt.assert_almost_equal(self.traveling.params['f'], np.array([5.912, 5.000]), decimal=3)

        # Test r_0, r_x1, r_x2 - Interim parameter for near field temperature 
        self.traveling._calc_interim_parameters_for_near_field_temp()
        npt.assert_almost_equal(self.traveling.params['r_0'], np.array([1.116, 1.116]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['r_x1'], np.array([0.000, 0.000]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['r_x2'], np.array([2.500, 2.500]), decimal=3)

        # Test T_nf - Average near field temperature
        self.traveling._calc_average_near_field_temp()
        npt.assert_almost_equal(self.traveling.params['T_nf'], np.array([1118.458, 1200.000]), decimal=3)

        # Test max_gas_temp - Max gas temperature
        self.traveling._define_max_gas_temperature()
        npt.assert_almost_equal(self.traveling.params['max_gas_temp'], np.array([1118.458, 1200.000]), decimal=3)


        """ FURTHER WORK """

        # Test amend parameters function
        # Test get exposure function
        # Test get time temperature curves function
        # Check temperatures with as a function of relative position
        # Far field temperatures?

if __name__ == '__main__':
    unittest.main()