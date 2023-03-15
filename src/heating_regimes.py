import numpy as np
import pandas as pd


class GenericRegime:
    REQUIRED_PARAMS = ['all required parameters']
    NAME = 'Some name'
    DESCRIPTION = 'Some description'

    def __init__(self, design_fire_inputs, crit_value):
        self.crit_value = crit_value
        self.relevent_df_indices = None
        self.params = {}

        self._get_relevant_design_fire_indices(design_fire_inputs)
        self._get_parameters(design_fire_inputs)

    def _get_relevant_design_fire_indices(self, design_fire_inputs):
        raise NotImplemented

    def _get_parameters(self, design_fire_inputs):
        raise NotImplemented

    def perform_initial_calculations(self):
        raise NotImplemented

    def check_bad_samples(self):
        raise NotImplemented

    def summarise_parameters(self):
        raise NotImplemented

    def get_exposure(self):
        raise NotImplemented


class FlashEC1(GenericRegime):

    REQUIRED_PARAMS = ['A_c', 'c_ratio', 'h_c', 'w_frac', 'h_w_eq', 'remain_frac', 'q_f_d', 't_lim', 'fabr_inrt']
    NAME = 'Flashover BS EN 1991-1-2'
    DESCRIPTION = 'Some description'

    def __init__(self, design_fire_inputs, crit_value, Of_limits):
        super().__init__(design_fire_inputs, crit_value)
        self.Of_limits = Of_limits

    def _get_relevant_design_fire_indices(self, design_fire_inputs):
        """Samples only relevant data from design fires based on criteria
        UNIT TEST REQUIRED"""
        # Get indeces
        self.relevent_df_indices = design_fire_inputs['A_c'] < self.crit_value

    def _get_parameters(self, design_fire_inputs):
        """Samples only revenat data from design fires based on criteria. UNITE TEST REQUIRED"""
        for param in FlashEC1.REQUIRED_PARAMS:
            try:
                self.params[param] = design_fire_inputs[param][self.relevent_df_indices]
            except KeyError:
                print(f'Missing input parameter for {FlashEC1.NAME} methodology: {param}')
                raise KeyError



    def perform_initial_calculations(self):
        """Performs all necessary time independent calculations needed for generation of time temperature curves
        Purpose of this method is to perform the calcs once at the initiation of the class so that the data
        can be reused for subsequent calls.
        Refer to the guidance of BS EN 1991-1-2 Appendix A"""

        self._calc_comp_sides()
        self._calc_perimeter()
        self._calc_total_area_enclosure()  # calc At
        self._calc_total_ventilation_area()  # calc Av
        self._calc_max_open_factor()  # calc Of_max
        self._calc_open_factor_breakage()  # calc Of
        self._apply_open_factor_limits()
        self._calc_GA_factor()  # calc Ga
        self._calc_total_surface_area_fuel_density() # calc q_t_d from q_f_d
        self._calc_t_max_vent()  # Calc t_max for ventilation controlled fire
        self._calc_t_max_fuel()  # Calc t_max for fuel control fire
        self._calc_open_factor_fuel() # Calc open factor for fuel controlled - Of lim
        self._calc_GA_lim_factor()  # Calc GA_lim
        self._calc_GA_lim_k_mod()  # Calc k factor for GA_lim
        self._define_burning_regime()  # Defines whether it is ventilation control or fuel control fire
        self._calc_max_temp_time()  # Calculates the time of max temperature
        self._calc_t_star_max()  # Calculates t star helping parameters
        self._calc_max_temp()  # calculates max temperature
        self._calc_fire_duration()  # Calculates burnout in [min]


    def _calc_comp_sides(self):
        """Calculates short and long side of compartment. UNIT TEST REQUIRED"""
        self.params['c_long'] = np.sqrt(self.params['A_c']/self.params['c_ratio'])
        self.params['c_short'] = self.params['c_ratio']*self.params['c_long']

    def _calc_perimeter(self):
        """UNITE TEST REQUIRED"""
        self.params['c_perim'] = 2*(self.params['c_long'] + self.params['c_short'])

    def _calc_total_area_enclosure(self):
        """UNITE TEST REQUIRED"""
        self.params['A_t'] = 2*self.params['A_c'] + self.params['c_perim']*self.params['h_c']

    def _calc_total_ventilation_area(self):
        """UNIT TEST REQUIRED"""
        self.params['A_v'] = self.params['w_frac']*self.params['c_perim']*self.params['h_w_eq']

    def _calc_max_open_factor(self):
        """See BS EN 1991-1-2 A.2a. UNIT TEST REQUIRED"""
        self.params['Of_max'] = self.params['A_v']*np.sqrt(self.params['h_w_eq'])/self.params['A_t']

    def _calc_open_factor_breakage(self):
        """Refer to TGN B4.5.3 and JCSS - 2 Clause 2.20.4.1. UNIT TEST"""
        self.params['Of'] = self.params['Of_max']*(1 - self.params['remain_frac'])

    def _apply_open_factor_limits(self):
        """Applies limits to Of for EC1 methodology which are user defined.
        See BS EN 1991-1-2 A.2a and  PD 6688-1-2:2007 Section 3.1.2(d)
        UNIT TEST REQUIRED"""
        self.params['Of'][self.params['Of'] > self.Of_limits[1]] = self.Of_limits[1]
        self.params['Of'][self.params['Of'] < self.Of_limits[0]] = self.Of_limits[0]

    def _calc_total_surface_area_fuel_density(self):
        """See BS EN 1993-1-2 A.7 UNIT TEST REQUIRED"""
        self.params['q_t_d'] = self.params['q_f_d']*self.params['A_c']/self.params['A_t']
        #Limits on q-t_d applied in accordance with BS EN 1991-1-2 Annex A (7)
        self.params['q_t_d'][self.params['q_t_d'] > 1000] = 1000
        self.params['q_t_d'][self.params['q_t_d'] < 50] = 50

    def _calc_open_factor_fuel(self):
        """See BS EN 1993-1-2 A.2a UNIT TEST REQUIRED"""
        self.params['Of_lim'] = 0.0001*self.params['q_t_d']/self.params['t_max_fuel']

    def _calc_GA_factor(self):
        """See BS EN 1993-1-2 A.9 UNIT TEST REQUIRED"""
        self.params['GA'] = ((self.params['Of']/self.params['fabr_inrt'])/(0.04/1160))**2

    def _calc_GA_lim_factor(self):
        """See BS EN 1993-1-2 A.8 UNIT TEST REQUIRED"""
        self.params['GA_lim'] = ((self.params['Of_lim']/self.params['fabr_inrt'])/(0.04/1160))**2

    def _calc_GA_lim_k_mod(self):
        """See BS EN 1993-1-2 A.10 UNIT TEST REQUIRED"""
        self.params['k'] = np.ones_like(self.params['GA_lim'])
        #Apply criteria from BS EN 1991-1-2 A.9
        crit = (self.params['Of'] > 0.04) & (self.params['q_t_d'] < 75) & (self.params['fabr_inrt'] < 1160)
        self.params['k'][crit] = 1 + ((self.params['Of'][crit]-0.04)/0.04) * ((self.params['q_t_d'][crit]-75)/75) * ((1160 - self.params['fabr_inrt'][crit])/1160)
        self.params['GA_lim'] = self.params['GA_lim']*self.params['k']

    def _calc_t_max_vent(self):
        """Maximum time for ventilation controlled fire. See BS EN 1993-1-2 A.7, UNIT TEST REQUIRED"""
        self.params['t_max_vent'] = 0.0002*self.params['q_t_d']/self.params['Of']

    def _calc_t_max_fuel(self):
        self.params['t_max_fuel'] = self.params['t_lim']/60

    def _define_burning_regime(self):
        """Decides whether burning regime is ventilation or fuel controlled, UNIT TEST REQUIRED"""
        self.params['regime'] = np.full(len(self.params['A_c']), 'V')
        self.params['regime'][self.params['t_max_fuel'] > self.params['t_max_vent']] = 'F'

    def _calc_max_temp_time(self):
        """Returns maximum time based from the two regimes. See BS EN 1991-1-2 A.7"""
        self.params['max_temp_t'] = np.max([self.params['t_max_fuel'], self.params['t_max_vent']], axis=0)

    def _calc_t_star_max(self):
        """Calculates t star max. UNIT TEST REQUIRED"""
        crit = self.params['regime'] == 'V'
        self.params['t_str_max_heat'] = np.full(len(self.params['A_c']), -1, dtype=np.float64)
        self.params['t_str_max_heat'][crit] = self.params['max_temp_t'][crit] * self.params['GA'][crit]
        self.params['t_str_max_heat'][~crit] = self.params['max_temp_t'][~crit] * self.params['GA_lim'][~crit]

        #Create t_str_max_cool for the cooling phase. See BS EN 1991-1-2 A.12
        self.params['t_str_max_cool_fuel'] = self.params['t_max_fuel'] * self.params['GA']
        self.params['t_str_max_cool_vent'] = self.params['t_max_vent'] * self.params['GA']


    def _calc_heat_phase_temp(self, t_str_heat):
        """Calculates a heating phase temperature for input effective time according to BS EN 1993-1-2 A.1
        UNIT TEST REQUIRED
        Inputs:
            t_str (array_like) - effective time calculated according to BS EN 1993-1-2 Anenx A
        Returns
            T (array like) - Gas temperature in [degC] """
        #TODO coordinate time units

        return 20 + 1325 * (1 - 0.324 * np.exp(-0.2 * t_str_heat) - 0.204 * np.exp(-1.7 * t_str_heat) - 0.472 * np.exp(-19 * t_str_heat))

    def _calc_max_temp(self):
        """Calculates max temp based on t star max using BS EN 1993-1-2 A.1
        UNIT TEST REQUIRED"""
        self.params['max_temp'] = self._calc_heat_phase_temp(self.params['t_str_max_heat'])

    def _calc_burnout(self):
        """Calculates burnout time based on the input parameters"""
        pass

    def _calc_cooling_phase_temp(self, t):
        """Calculates cooling phase temperatures in accordance with BS EN 1991-1-2 Annex A (11)

        Inputs:
            t (array like): time vector in [min - TBC]
        Returns:
            Array of temperatures for particular times in [degC]"""
        #TODO coordinate time units

        cool_temp = np.full(len(self.params['A_c']), -1, dtype=np.float64)
        t_str_cool = self.params['GA']*t/60

        # Case A.11a
        crit = (self.params['t_str_max_cool_vent'] <= 0.5) & (self.params['regime'] == 'V')
        cool_temp[crit] = self.params['max_temp'][crit] - 625 * (t_str_cool[crit] - self.params['t_str_max_cool_vent'][crit])
        crit = (self.params['t_str_max_cool_vent'] <= 0.5) & (self.params['regime'] == 'F')
        cool_temp[crit] = self.params['max_temp'][crit] - 625 * (t_str_cool[crit] - self.params['t_str_max_cool_fuel'][crit])
        # Case A.11b
        crit = (self.params['t_str_max_cool_vent'] > 0.5) & (self.params['t_str_max_cool_vent'] < 2) & (self.params['regime'] == 'V')
        cool_temp[crit] = self.params['max_temp'][crit] - 250 * (3 - self.params['t_str_max_cool_vent'][crit]) * (t_str_cool[crit] - self.params['t_str_max_cool_vent'][crit])
        crit = (self.params['t_str_max_cool_vent'] > 0.5) & (self.params['t_str_max_cool_vent'] < 2) & (self.params['regime'] == 'F')
        cool_temp[crit] = self.params['max_temp'][crit] - 250 * (3 - self.params['t_str_max_cool_vent'][crit]) * (t_str_cool[crit] - self.params['t_str_max_cool_fuel'][crit])
        # Case A.11c
        crit = (self.params['t_str_max_cool_vent'] >= 2) & (self.params['regime'] == 'V')
        cool_temp[crit] = self.params['max_temp'][crit] - 250 * (t_str_cool[crit] - self.params['t_str_max_cool_vent'][crit])
        crit = (self.params['t_str_max_cool_vent'] >= 2) & (self.params['regime'] == 'F')
        cool_temp[crit] = self.params['max_temp'][crit] - 250 * (t_str_cool[crit] - self.params['t_str_max_cool_fuel'][crit])

        #Temperature cannot go below ambient
        cool_temp[cool_temp < 20] = 20

        return cool_temp

    def _calc_fire_duration(self):
        """Estimates total burnout time based on heating phase time and estimates for decay duration based on
        reworking of equations in BS EN 1991-1-2 Annex A (11)"""

        a = np.full(len(self.params['A_c']), -1, dtype=np.float64)  # holds slopes of the cooling phase for diff. fires

        crit = (self.params['t_str_max_cool_vent'] <= 0.5) & (self.params['regime'] == 'V')
        a[crit] = 625
        crit = (self.params['t_str_max_cool_vent'] <= 0.5) & (self.params['regime'] == 'F')
        a[crit] = 625

        crit = (self.params['t_str_max_cool_vent'] > 0.5) & (self.params['t_str_max_cool_vent'] < 2) & (self.params['regime'] == 'V')
        a[crit] = 250 * (3 - self.params['t_str_max_cool_vent'][crit])
        crit = (self.params['t_str_max_cool_vent'] > 0.5) & (self.params['t_str_max_cool_vent'] < 2) & (self.params['regime'] == 'F')
        a[crit] = 250 * (3 - self.params['t_str_max_cool_vent'][crit])

        crit = (self.params['t_str_max_cool_vent'] >= 2) & (self.params['regime'] == 'V')
        a[crit] = 250
        crit = (self.params['t_str_max_cool_vent'] >= 2) & (self.params['regime'] == 'F')
        a[crit] = 250

        self.params['burnout'] = (self.params['max_temp_t'] + (self.params['max_temp'] - 20)/a/self.params['GA'])*60


    def get_exposure(self, t):
        """Produce a vector of exposure temperatures at specific t. UNIT TEST REQUIRED"""

        t_str_heat = np.full(len(self.params['A_c']), -1, dtype=np.float64)

        # Calculate t_str_heat based on fire regime:
        crit = self.params['regime'] == 'V'
        t_str_heat[crit] = self.params['GA'][crit] * t/60
        crit = self.params['regime'] == 'F'
        t_str_heat[crit] = self.params['GA_lim'] [crit] * t/60

        #Calculate heating and colling temperatures. These are compared to avoid discontinuities
        # The logic is that if t_str is smaller than t_max_str the resulting temp will allways be bigger
        #than temp max calculated from the heating phase Similar approach is implemented in the SFE toolkit
        heat_phase_temp = self._calc_heat_phase_temp(t_str_heat)
        cool_phase_temp = self._calc_cooling_phase_temp(t)

        return np.min([heat_phase_temp, cool_phase_temp], axis=0)

    def get_time_temperature_curves(self, t_values):
        """Get time temperature curves"""

        # TODO Rewrite this to allow for subsampling and get_exposure method to allow subsampling
        curve_data = np.full((len(t_values), len(self.params['A_c'])), -1, dtype=np.float64)
        for i, t in enumerate(t_values):
            curve_data[i, :] = self.get_exposure(t)
        return curve_data


    def summarise_parameters(self):
        """Returns all calculated parameters in human readable table format"""
        data = pd.DataFrame.from_dict(self.params)
        col_list = ['c_ratio', 'c_long', 'c_short', 'A_c', 'h_c', 'c_perim', 'A_t', 'h_w_eq', 'w_frac', 'remain_frac',
                    'A_v', 'Of_max', 'Of', 'fabr_inrt', 'GA', 'q_f_d', 'q_t_d', 't_max_vent', 't_lim', 't_max_fuel',
                    'Of_lim', 'k', 'GA_lim', 'regime', 'max_temp_t', 't_str_max_heat', 't_str_max_cool_vent', 't_str_max_cool_fuel',
                    'max_temp', 'burnout']
        data = data[col_list]
        return data

    def check_bad_samples(self):
        """Discards/amends  bad samples which produce unphysical results due imperfections of the adopted
         methodology TODO this to be updated following statistical testing of outputs """

        raise NotImplemented


class TravelingISO16733(GenericRegime):

    REQUIRED_PARAMS = ['A_c', 'h_c', 'c_ratio', 'q_f_d', 'Q', 'spr_rate', 'flap_angle', 'T_nf_max', 'T_amb']
    NAME = 'Traveling ISO 16733-2'
    DESCRIPTION = 'Some description'

    def __init__(self,design_fire_inputs, crit_value, assess_loc, max_travel_path):
        super().__init__(design_fire_inputs, crit_value)
        self.assess_loc = assess_loc  # Location of assessment as a fraction of travel path
        self.max_travel_path = max_travel_path  # Location of assessment as a fraction of travel path

    def _get_relevant_design_fire_indices(self, design_fire_inputs):
        """Samples only relevant data from design fires based on criteria
        UNIT TEST REQUIRED"""
        # Get indeces
        self.relevent_df_indices = design_fire_inputs['A_c'] > self.crit_value

    def _get_parameters(self, design_fire_inputs):
        """Samples only relevant data from design fires based on criteria. UNITE TEST REQUIRED"""
        for param in TravelingISO16733.REQUIRED_PARAMS:
            try:
                self.params[param] = design_fire_inputs[param][self.relevent_df_indices]
            except KeyError:
                print(f'Missing input parameter for {TravelingISO16733.NAME} methodology: {param}')
                raise KeyError

    def _calc_comp_sides(self):
        """Calculates short and long side of compartment. UNIT TEST REQUIRED"""
        self.params['c_long'] = np.sqrt(self.params['A_c']/self.params['c_ratio'])
        #  Checks if max traveling path is more than the maximum one possible for the floor plate.
        #  If true the maximum is assigned. SHort side is computed based on the side ratio.
        self.params['c_long'][self.params['c_long'] > self.max_travel_path] = self.max_travel_path
        self.params['c_short'] = self.params['c_ratio']*self.params['c_long']

    def _calc_assess_loc(self):
        """Calculations the position along the fire path where the time temeprature curve is to be assessed"""
        self.params['x_loc'] =self.assess_loc*self.params['c_long']

    def _calc_burning_time(self):
        """Calculates burning time for individual segment in [s]. See TGN C2"""
        self.params['t_b'] = 1000*self.params['q_f_d']/self.params['Q']

    def _calc_fire_base_length(self):
        """Calculates burning time for individual segment. See TGN C2"""
        self.params['L_f'] = self.params['spr_rate'] * self.params['t_b']/1000

    def _calc_fire_base_area(self):
        """Calculates fire base area. See TGN C2 - p.C3"""
        self.params['A_f'] = self.params['L_f']*self.params['c_short']

    def _calc_relative_fire_size(self):
        """Calculates relative fire size - ratio of fire base to total length of the fire path"""
        self.params['L_str'] = self.params['L_f']/self.params['c_long']

    def _calc_burnout(self):
        """Calculates burning time for the whole traveling path in [min]. See TGN C2"""
        self.params['burnout'] = 1000 * (self.params['c_long'] + self.params['L_f'])/self.params['spr_rate']/60

    def _calc_flap_l(self):
        """Calculates flapping length [m] based on flapping angle. See TGN C2"""
        self.params['f'] = self.params['L_f'] + 2*self.params['h_c']*np.tan(np.radians(self.params['flap_angle']))

    def _calc_interim_parameters_for_near_field_temp(self):
        """Method estimates r_x1, r_x2, and r_0 used for calculating averaged near field temperature.
        See TGN C2 p. C3"""

        self.params['r_0'] = self.params['Q'] * self.params['A_f'] * (5.38 / (self.params['h_c'] * (self.params['T_nf_max'] - self.params['T_amb']))) ** (3 / 2)
        self.params['r_x1'] = np.max([np.zeros(len(self.params['A_c'])), self.params['r_0'] - 0.5 * self.params['L_f']],axis=0)
        self.params['r_x2'] = np.max([0.5 * self.params['L_f'], self.params['r_0']], axis=0)

    def _calc_average_near_field_temp(self):
        """Estimates average near field temperature considering flapping angle and heat release rate.
         See TGN C2 p. C3"""

        #For inputs when the flap angle is 0 T_nf is equal to T_nf_max. Else use TGN equation on p. C3.
        self.params['T_nf'] = np.full(len(self.params['A_c']), -1, dtype=np.float64)
        crit = self.params['flap_angle'] < 0.001
        self.params['T_nf'][crit] = self.params['T_nf_max'][crit]

        # Get short values for case where flapping angle != 0
        T_amb = self.params['T_amb'][~crit]
        T_nf_max = self.params['T_nf_max'][~crit]
        Q = self.params['Q'][~crit]
        A_f = self.params['A_f'][~crit]
        r_x1 = self.params['r_x1'][~crit]
        r_x2 = self.params['r_x2'][~crit]
        f = self.params['f'][~crit]
        h_c = self.params['h_c'][~crit]
        L_f = self.params['L_f'][~crit]
        self.params['T_nf'][~crit] = T_amb + (T_nf_max * (2*r_x1+L_f) - 2*T_amb*r_x2)/f + (32.28*(Q*A_f)**(2/3) * ((0.5*f)**(1/3) - r_x2**(1/3)))/(f*h_c)

    def get_exposure(self, t, x_rel_loc=None):
        """Gets temperature from traveling fire exposure at specific location, x_loc, and specific time t
        Inputs:
            t (float): calculation time in [min] - TBC
            x_rel_loc (float): Relative location of assessment point. It must be between 0 and 1. If None then ass_loc value used
            Defaults to None
        Returns:
            T_exp(array like): Vector of exposure temperatures with size N.
        """

        if x_rel_loc is not None:
            self.params['x_loc'] = x_rel_loc * self.params['c_long']

        x_str = self.params['spr_rate']*t*60/1000
        crit = x_str > self.params['c_long']

        x_str_t = self.params['c_long'].copy()
        x_str_t[~crit] = x_str[~crit]
        # Calculate L_str
        L_str_t = np.full(len(self.params['A_c']), -1, dtype=np.float64)
        L_str_t[crit] = np.max([1 + (self.params['L_f'][crit] - x_str[crit])/self.params['c_long'][crit], np.zeros(len(self.params['A_c'][crit]))], axis=0)
        L_str_t[~crit] = np.min([self.params['L_str'][~crit], x_str[~crit]/self.params['c_long'][~crit]], axis=0)

        #Calculate  position condition
        dist = np.absolute(self.params['x_loc'] + 0.5*L_str_t*self.params['c_long'] - x_str_t)
        crit = dist > 0.5*self.params['L_f']

        #Calculate temperature
        T_exp = np.zeros(len(self.params['A_c']), dtype=np.float64)
        T_exp[crit] = self.params['T_amb'][crit] + ((5.38/self.params['h_c']) * (L_str_t*self.params['c_long']*self.params['c_short']*self.params['Q']/dist)**(2/3))[crit]
        T_exp[~crit] = self.params['T_nf'][~crit]
        T_exp[T_exp > self.params['T_nf']] = self.params['T_nf'][T_exp > self.params['T_nf']]

        return T_exp

    def get_time_temperature_curves(self, t_values, x_rel_loc=None):
        """Get time temperature curves for defined times and relative locations
        Inputs:
            x_rel_loc (float): Relative location of assessment point. It must be between 0 and 1. If None then ass_loc value used
            Defaults to None
        Returns:
            curve_data (array like): Array of exposure temperatures with shape len(t_values) x N
            """

        # TODO Rewrite this to allow for subsampling and get_exposure method to allow subsampling
        curve_data = np.full((len(t_values), len(self.params['A_c'])), -1, dtype=np.float64)
        for i, t in enumerate(t_values):
            curve_data[i, :] = self.get_exposure(t, x_rel_loc)
        return curve_data

    def perform_initial_calculations(self):
        self._calc_comp_sides()
        self._calc_burning_time()
        self._calc_fire_base_length()
        self._calc_assess_loc()
        self._calc_fire_base_area()
        self._calc_relative_fire_size()
        self._calc_burnout()
        self._calc_flap_l()
        self._calc_interim_parameters_for_near_field_temp()
        self._calc_average_near_field_temp()

    def summarise_parameters(self):
        """Returns all calculated parameters in human readable table format"""
        data = pd.DataFrame.from_dict(self.params)
        col_list = ['c_long', 'c_ratio', 'c_short', 'A_c', 'h_c', 'q_f_d', 'Q', 'spr_rate', 'T_amb', 'flap_angle',
                    't_b', 'L_f', 'x_loc', 'A_f', 'L_str', 'f', 'r_0', 'r_x1', 'r_x2', 'T_nf_max', 'T_nf', 'burnout']
        data = data[col_list]
        return data

    def check_bad_samples(self):
        """Discards/amends  bad samples which produce unphysical results due imperfections of the adopted
         methodology TODO this to be updated following statistical testing of outputs """

        raise NotImplemented