import unittest
import numpy as np
import numpy.testing as npt
import src.heating_regimes as hr


class TestUniEC1(unittest.TestCase):

    def setUp(self):
        inputs = {
            'A_c': np.array([40, 60, 1000, 100, 20]),
            'c_ratio': np.array([0.2, 1, 0, 1, 1]),
            'h_c': np.array([3, 4, 0, 4, 4]),
            'w_frac': np.array([0.15, 0.5, 0, 0.5, 0.04]),
            'remain_frac': np.array([0.0, 0.1, 0, 0, 0]),
            'q_f_d': np.array([300, 300, 0, 1000, 1000]),
            't_lim': np.array([15, 20, 0, 25, 25]),
            'w_frac_h': np.array([0.5, 0.75, 0, 1, 0.5]),
            'fabr_inrt' : np.array([1000, 1000, 0, 300, 500]),
            'foo': np.array([1000, 1000, 0, 0, 0])}
        crit_value = 100.1
        Of_limits = [0.02, 0.2]
        max_fire_duration = 1000

        self.parametric = hr.UniEC1(
            design_fire_inputs=inputs,
            crit_value=crit_value,
            max_fire_duration=max_fire_duration,
            Of_limits=Of_limits)

    def test_get_relevant_design_fire_indeces(self):
        npt.assert_equal(self.parametric.relevent_df_indices, np.array([0, 1, 3, 4]))
        self.assertFalse(self.parametric.is_empty, 'Issue with empty method definition')

    def test_get_parameters(self):
        self.assertFalse('foo' in self.parametric.params)
        self.assertEqual(len(self.parametric.params), 9)

    def test_perform_initial_calculations(self):

        # Test _calc_comp_sides calculations
        self.parametric._calc_comp_sides()
        npt.assert_almost_equal(self.parametric.params['c_long'], np.array([14.142, 7.746, 10.000, 4.472]), decimal=3)
        npt.assert_almost_equal(self.parametric.params['c_short'], np.array([2.828, 7.746, 10.000, 4.472]), decimal=3)

        # Test c_perim
        self.parametric._calc_perimeter()
        npt.assert_almost_equal(self.parametric.params['c_perim'], np.array([33.941, 30.984, 40.000, 17.889]), decimal=3)

        # Test A_t - total enclosure area
        self.parametric._calc_total_area_enclosure()
        npt.assert_almost_equal(self.parametric.params['A_t'], np.array([181.823, 243.935, 360.000, 111.554]), decimal=3)

        # Test h_w_eq - average window height
        self.parametric._calc_average_window_height()
        npt.assert_almost_equal(self.parametric.params['h_w_eq'], np.array([1.500, 3.000, 4.000, 2.000]), decimal=3)

        # Test A_v - total ventilation area
        self.parametric._calc_total_ventilation_area()
        npt.assert_almost_equal(self.parametric.params['A_v'], np.array([7.637, 46.476, 80.000, 1.431]), decimal=3)

        # Test Of - breakage opening factor
        self.parametric._calc_max_open_factor()
        self.parametric._calc_open_factor_breakage()
        npt.assert_almost_equal(self.parametric.params['Of'], np.array([0.051, 0.297, 0.444, 0.018]), decimal=3)

        # Test apply open factor limits
        self.parametric._apply_open_factor_limits()
        npt.assert_almost_equal(
            self.parametric.params['Of'], np.array([0.051, self.parametric.Of_limits[1], self.parametric.Of_limits[1], self.parametric.Of_limits[0]]), decimal=3)

        # Test GA - test calc Ga
        self.parametric._calc_GA_factor()
        npt.assert_almost_equal(self.parametric.params['GA'], np.array([2.225, 33.640, 373.778, 1.346]), decimal=3)

        # Test q_t_d - total surface area fuel density
        self.parametric._calc_total_surface_area_fuel_density()
        npt.assert_almost_equal(self.parametric.params['q_t_d'], np.array([65.998, 73.790, 277.778, 179.285]), decimal=3)

        # Test t_max_vent - time to max temp for vent controlled fire
        self.parametric._calc_t_max_vent()
        npt.assert_almost_equal(self.parametric.params['t_max_vent'], np.array([0.257, 0.074, 0.278, 1.793]), decimal=3)

        # Test t_max_fuel - time to max temp for fuel controlled fire
        self.parametric._calc_t_max_fuel()
        npt.assert_almost_equal(self.parametric.params['t_max_fuel'], np.array([0.250, 0.333, 0.417, 0.417]), decimal=3)

        # Test Of_lim - Open factor for fuel controlled
        self.parametric._calc_open_factor_fuel()
        npt.assert_almost_equal(self.parametric.params['Of_lim'],  np.array([0.026, 0.022, 0.067, 0.043]), decimal=3)

        # Test k and Test GA_lim - K factor for GA_lim - NOTE: Ga_lim is modified with k
        self.parametric._calc_GA_lim_factor()
        self.parametric._calc_GA_lim_k_mod()
        npt.assert_almost_equal(self.parametric.params['k'], np.array([0.995, 0.991, 1.000, 1.000]), decimal=3)
        npt.assert_almost_equal(self.parametric.params['GA_lim'], np.array([0.583, 0.408, 41.531, 6.228]), decimal=3)

        # Test burning regime - checking for fuel or vent control
        self.parametric._define_burning_regime()
        npt.assert_equal(self.parametric.params['regime'], np.array(['V', 'F', 'F', 'V']))

        # Test max_temp_t - time of maximum temperature 
        self.parametric._calc_max_temp_time()
        npt.assert_almost_equal(self.parametric.params['max_temp_t'], np.array([0.257, 0.333, 0.417, 1.793]), decimal=3)

        # Test t_star_max
        self.parametric._calc_t_star_max()
        npt.assert_almost_equal(self.parametric.params['t_str_max_heat'], np.array([0.571, 0.136, 17.305, 2.412]), decimal=3)

        # Test max_temp - maximum temperature
        self.parametric._calc_max_temp()
        npt.assert_almost_equal(self.parametric.params['max_temp'], np.array([859.632, 665.721, 1331.519, 1075.543]), decimal=3)

        # Test fire duration - fire duration
        self.parametric._calc_fire_duration()
        npt.assert_almost_equal(self.parametric.params['burnout'], np.array([52.676, 24.607, 25.842, 295.837]), decimal=3)

        # Test _define_max_gas_temp - for external class use
        self.parametric._define_max_gas_temp()
        npt.assert_almost_equal(self.parametric.params['max_gas_temp'], np.array([859.632, 665.721, 1331.519, 1075.543]), decimal=3)

    def test_get_exposure(self):
        self.parametric.perform_initial_calculations()

        # Test t=0 min
        answ = self.parametric.get_exposure(t=0, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20]), decimal=3)
        # Test t=20 min
        answ = self.parametric.get_exposure(t=20, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([755.939, 665.721, 1318.065, 826.316]), decimal=3)
        # Test t=40 min
        answ = self.parametric.get_exposure(t=40, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([305.49, 20, 20, 927.386]), decimal=3)
        # Test t=400 min
        answ = self.parametric.get_exposure(t=400, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20]), decimal=3)

        # Test subsample masking
        answ = self.parametric.get_exposure(t=40, subsample_mask=[True, False, False, True])
        npt.assert_almost_equal(answ, np.array([305.49, 927.386]), decimal=3)


class TestTravelingISO16733(unittest.TestCase):

    def setUp(self):
        inputs = {
            'A_c': np.array([20, 500, 1800, 5000, 1200, 1600]),
            'h_c': np.array([4, 4, 3, 4.5, 4.5, 4.5]),
            'c_ratio': np.array([0.8, 0.8, 0.5, 0.5, 0.75, 0.25]),
            'q_f_d': np.array([570, 570, 150, 600, 400, 400]),
            'Q': np.array([290, 290, 550, 150, 150, 150]),
            'spr_rate': np.array([2.544, 2.544, 10, 1.5, 16,1]),
            'flap_angle': np.array([6.5, 6.5, 5, 6.5, 5, 0]),
            'T_nf_max': np.array([1200, 1200, 1200, 1100, 1100, 1200]),
            'T_amb': np.array([20, 20, 20, 20, 20, 20]),
            'foo': np.array([20, 20, 20, 20, 20, 20])}
        crit_value = 100
        assess_loc = 0.8
        max_travel_path = 80
        max_fire_duration = 1000

        self.traveling = hr.TravelingISO16733(
            design_fire_inputs=inputs,
            crit_value=crit_value,
            assess_loc=assess_loc,
            max_travel_path=max_travel_path,
            max_fire_duration=max_fire_duration)

    def test_get_relevant_design_fire_indeces(self):
        npt.assert_equal(self.traveling.relevent_df_indices, np.array([1, 2, 3, 4, 5]))
        self.assertFalse(self.traveling.is_empty, 'Issue with empty method definition')

    def test_get_parameters(self):
        self.assertFalse('foo' in self.traveling.params)
        self.assertEqual(len(self.traveling.params), 9)

    def test_perform_initial_calculations(self):

        # Test c_long and c_short
        self.traveling._calc_comp_sides()
        npt.assert_almost_equal(self.traveling.params['c_long'], np.array([25.000, 60.000, 80.000, 40.000, 80.000]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['c_short'], np.array([20.000, 30.000, 40.000, 30.000, 20.000]), decimal=3)
        
        # Test x_loc - position along fire path
        self.traveling._calc_assess_loc()
        npt.assert_almost_equal(self.traveling.params['x_loc'], np.array([20.000, 48.000, 64.000, 32, 64.000]), decimal=3)

        # Test t_b - fire burning time 
        self.traveling._calc_burning_time()
        npt.assert_almost_equal(self.traveling.params['t_b'], np.array([1965.517, 272.727, 4000, 2666.667, 2666.667]), decimal=3)

        # Test L_f - fire base length 
        self.traveling._calc_fire_base_length()
        npt.assert_almost_equal(self.traveling.params['L_f'], np.array([5.000, 2.727, 6.000, 42.667, 2.667]), decimal=3)

        # Test A_f - fire base area 
        self.traveling._calc_fire_base_area()
        npt.assert_almost_equal(self.traveling.params['A_f'], np.array([100.006, 81.818, 240.000, 1200, 53.333]), decimal=3)
        
        # Test L_str - Relative fire size 
        self.traveling._calc_relative_fire_size()
        npt.assert_almost_equal(self.traveling.params['L_str'], np.array([0.200, 0.045, 0.075, 1.067, 0.033]), decimal=3)

        # Test burnout - burning time for whole travelling fire path 
        self.traveling._calc_burnout()
        npt.assert_almost_equal(self.traveling.params['burnout'], np.array([196.543, 104.545, 955.556, 86.111, 1377.778]), decimal=3)

        # Test f - flapping length
        self.traveling._calc_flap_l()
        npt.assert_almost_equal(self.traveling.params['f'], np.array([5.912, 3.252, 7.025, 43.454, 2.667]), decimal=3)

        # Test r_0, r_x1, r_x2 - Interim parameter for near field temperature 
        self.traveling._calc_interim_parameters_for_near_field_temp()
        npt.assert_almost_equal(self.traveling.params['r_0'], np.array([1.116, 2.666, 1.326, 6.630, 0.258]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['r_x1'], np.array([0.000, 0.262, 0, 0, 0]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['r_x2'], np.array([2.500, 1.626, 3, 21.333, 1.333]), decimal=3)

        # Test T_nf - Average near field temperature
        self.traveling._calc_average_near_field_temp()
        npt.assert_almost_equal(self.traveling.params['T_nf'], np.array([1118.458, 1200.000, 1029.063, 1089.354, 1200]), decimal=3)

        # Test max_gas_temp - Max gas temperature
        self.traveling._define_max_gas_temperature()
        npt.assert_almost_equal(self.traveling.params['max_gas_temp'], np.array([1118.458, 1200.000, 1029.063, 1089.354, 1200]), decimal=3)

    def test_amend_long_duration_fires(self):
        self.traveling.perform_initial_calculations()
        self.traveling._amend_long_duration_fires()

        self.assertEqual(self.traveling.amended_df_indices['long_duration'], 5)
        npt.assert_almost_equal(self.traveling.params['burnout'][4], np.array([self.traveling.max_fire_duration]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['c_long'][4], np.array([57.333]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['x_loc'][4], np.array([45.866]), decimal=3)
        npt.assert_almost_equal(self.traveling.params['L_str'][4], np.array([0.047]), decimal=3)

    def test_get_exposure(self):
        self.traveling.perform_initial_calculations()
        self.traveling.check_bad_samples()

        # Test t=0 min
        answ = self.traveling.get_exposure(t=0, x_rel_loc=0.1,  subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20, 20]), decimal=3)
        answ = self.traveling.get_exposure(t=0, x_rel_loc=0.5, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20, 20]), decimal=3)
        answ = self.traveling.get_exposure(t=0, x_rel_loc=0.9, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20, 20]), decimal=3)

        # Test t=100 min
        answ = self.traveling.get_exposure(t=100, x_rel_loc=0.1, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([288.821, 181.54, 1029.063, 20, 1200]), decimal=3)
        answ = self.traveling.get_exposure(t=100, x_rel_loc=0.5, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([1118.458, 262.395, 144.199, 20, 77.477]), decimal=3)
        answ = self.traveling.get_exposure(t=100, x_rel_loc=0.9, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([298.449, 835.989, 99.813, 20, 56.755]), decimal=3)

        # Test t=1001 min
        answ = self.traveling.get_exposure(t=1001, x_rel_loc=0.1, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20, 20]), decimal=3)
        answ = self.traveling.get_exposure(t=1001, x_rel_loc=0.5, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20, 20]), decimal=3)
        answ = self.traveling.get_exposure(t=1001, x_rel_loc=0.9, subsample_mask=None)
        npt.assert_almost_equal(answ, np.array([20, 20, 20, 20, 20]), decimal=3)

        # est t=100 min with subsample mask
        answ = self.traveling.get_exposure(t=100, x_rel_loc=0.5, subsample_mask=[True, False, False, False, True])
        npt.assert_almost_equal(answ, np.array([1118.458,  77.476]), decimal=3)

if __name__ == '__main__':
    unittest.main()