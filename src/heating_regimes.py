import numpy as np


class GenericRegime:
    REQUIRED_PARAMS = ['all required parameters']
    def __init__(self, design_fire_inputs, crit_value):
        self.crit_value = crit_value
        self.relevent_df_indices = None
        self.params = {}

        self._get_relevant_design_fire_indices(design_fire_inputs)
        self._get_parameters(design_fire_inputs)

    def get_exposure(self):
        raise NotImplemented

    def _get_relevant_design_fire_indices(self, design_fire_inputs):
        raise NotImplemented


class FlashEC1(GenericRegime):

    REQUIRED_PARAMS = ['A_c', 'c_ratio', 'h_c', 'w_frac', 'h_w_eq', 'break_frac', 'q_f_d', 't_lim', 'fabr_inrt']

    def __init__(self, design_fire_inputs, crit_value):
        super().__init__(design_fire_inputs, crit_value)


    def _get_relevant_design_fire_indices(self, design_fire_inputs):
        """Samples only relevant data from design fires based on criteria
        UNIT TEST REQUIRED"""
        # Get indeces
        self.relevent_df_indices = design_fire_inputs['A_c'] < self.crit_value

    def _get_parameters(self, design_fire_inputs):
        """Samples only revenat data from design fires based on criteria. UNITE TEST REQUIRED"""
        for param in FlashEC1.REQUIRED_PARAMS:
            self.params[param] = design_fire_inputs[param][self.relevent_df_indices]

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
        """UNITE TEST REQUIRED"""
        self.params['A_v'] = self.params['w_frac']*self.params['c_perim']*self.params['h_c']

    def _calc_max_open_factor(self):
        """See BS EN 1991-1-2 A.2a"""
        self.params['Of_max'] = self.params['A_v']*np.sqrt(self.params['h_w_eq'])/self.params['A_t']

    def _calc_open_factor_breakage(self):
        """Refer to TGN B4.5.3 and JCSS - 2 Clause 2.20.4.1"""
        self.params['Of'] = self.params['Of_max']*(1 - self.params['break_frac'])

    def _apply_open_factor_limits(self):
        """Applies limits to Of for EC1 methodology.
        See BS EN 1991-1-2 A.2a and  PD 6688-1-2:2007 Section 3.1.2(d)
        UNIT TEST REQUIRED"""
        self.params['Of'][self.params['Of'] >0.2] = 0.2
        self.params['Of'][self.params['Of'] < 0.01] = 0.01

    def _calc_total_surface_area_fuel_density(self):
        """See BS EN 1993-1-2 A.7 UNIT TEST REQUIRED"""
        self.params['q_t_d'] = self['q_f_d']*self.params['A_c']/self.params['A_t']
        #Limits on q-t_d applied in accordance with BS EN 1991-1-2 Annex A (7)
        self.params['q_t_d'][self.params['q_t_d'] > 1000] = 1000
        self.params['q_t_d'][self.params['q_t_d'] < 50] = 50

    def _calc_open_factor_fuel(self):
        """See BS EN 1993-1-2 A.2a UNIT TEST REQUIRED"""
        self.param['Of_lim'] = 0.0001*self.param['q_t_d']/self.param['t_max_fuel']

    def _calc_GA_factor(self):
        """See BS EN 1993-1-2 A.9 UNIT TEST REQUIRED"""
        self.params['GA'] = ((self.params['Of']/self.params['fabr_inrt'])/(0.04/1160))**2

    def _calc_GA_lim_factor(self):
        """See BS EN 1993-1-2 A.8 UNIT TEST REQUIRED"""
        self.params['GA_lim'] = ((self.params['Of_lim']/self.params['fabr_inrt'])/(0.04/1160))**2

    def _calc_GA_min_mod(self):
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


    def _prepare_input_params(self):
        """Gets only reveant input paraemters from the """
        pass
