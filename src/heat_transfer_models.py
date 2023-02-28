import numpy as np

import equivalent_curves as ecr

class GenericHT():
    """Generic class for heat transfer analysis"""

    equivalent_curves = {
        'ISO_834': ecr.StandardFire}

    def __init__(self, equivalent_curve, sect_prop, mat_prop, prot_prop, T_lim, eqv_max):
        self.label = 'Generic'
        self.descr = 'Generic descr'

        self.prot_prop = prot_prop
        self.mat_prop = mat_prop
        self.sect_prop = self._process_sample_section_geometry(sect_prop)

        self.T_lim = T_lim
        self.eqv_max = eqv_max
        self.ecr = self._load_equivalent_curve(equivalent_curve)
        self.equiv_protect = None

    def _load_equivalent_curve(self, equivalent_curve):
        return GenericHT.equivalent_curves[equivalent_curve]()

    def _process_sample_section_geometry(self, sect_prop):
        raise NotImplemented

    def get_equivelant_protection(self):
        raise NotImplemented

    def plot_equivelant_protection_curve(self):
        raise NotImplemented

    def calc_thermal_response(self):
        raise NotImplemented


class SteelEC3(GenericHT):

    def __init__(self, equivalent_curve, sect_prop, mat_prop, prot_prop, T_lim, eqv_max,dt):
        super().__init__(equivalent_curve, sect_prop, mat_prop, prot_prop, T_lim, eqv_max)
        self.label = 'Steel EC3 HT'
        self.descr = '1D heat transfer in accordance with BS EN 1993-1-2'
        self.dt = dt #time step size


    def _process_sample_section_geometry(self, sect_prop):
        """Return section factor which is needed for heat transfer calculation
        UNIT TEST REQUIRED.

        Inputs:
            sect_prop (dict) : Contains the following fields in [m]
            'B' - section breadth
            'D' - section debth
            't_fl' - flange thickness
            'wb_t' - web thickness
            'exp_sides' - either 'three' or 'fours'
            OR 'Av' - pre calculated section factor
        Returns:
            dict: Containing section factor, 'Av' in [1/m] """

        if 'Av' in sect_prop:
            return {'Av': sect_prop['Av']}

        #calculate area
        A = 2*sect_prop['fl_t']*sect_prop['B'] + (sect_prop['D'] - 2*sect_prop['fl_t'])*sect_prop['wb_t']
        #calculate heated perimeter for either 4 or 3 side exposure
        if sect_prop['exp_sides'] == 'four':
            P = 2*sect_prop['D'] + 4*sect_prop['B'] - 2*sect_prop['wb_t']
        elif sect_prop['exp_sides'] == 'three':
            P = 2 * sect_prop['D'] + 3 * sect_prop['B'] - 2 * sect_prop['wb_t']
        return {'Av': P/A}


    def get_equivelant_protection(self):
        """Calculates an interpolation curve of required fire protection material for equivalent protection
        thickness based on given gas temperature curve, section, material, and protection material properties.
         UNIT TEST REQUIRED

         Inputs:
            asd
        Returns:
            asd

         """

        T_ini = self.ecr.get_temp(0) #Get the initial temperature to equal to the gas temp. at t=0

        #Get array of protection thicknesses

        #calculate fi

        #start looping through Forward Euler Method over times tps

        #calculate delta T as per BS EN 1993-1-2:2005 Clause 4.2.5.2
        #update T

        #Interpolate for limiting temperature

    @staticmethod
    def _calc_steel_hc(T_m):
        """Vectorised calculation for steel heat capacity in accordance with BS EN 1993-1-2 Clause 3.4.1.2
        UNIT TEST REQUIRED

        Inputs:
            T (flor or array like) : temperature in [degC]"""

        if not isinstance(T_m, np.ndarray):
            T_m = np.asarray(T_m, dtype=np.float64)

        c_a = np.zeros_like(T_m)
        if np.any(T_m < 20):
            print('WARNING - Member temperature less than 20 degC. Outside definitions for steel heat capacity')
        idx = T_m < 20
        c_a[idx] = 425 + 0.773*T_m[idx] - 0.00169*T_m[idx]**2 + 0.00000222*T_m[idx]**3

        idx = (T_m >= 20) & (T_m < 600)
        c_a[idx] = 425 + 0.773 * T_m[idx] - 0.00169 * T_m[idx] ** 2 + 0.00000222 * T_m[idx] ** 3

        idx = (T_m >= 600) & (T_m < 735)
        c_a[idx] = 666 + 13002 / (738 - T_m[idx])

        idx = (T_m >= 735) & (T_m < 900)
        c_a[idx] = 545 + 17820 / (T_m[idx] - 731)

        idx = (T_m >= 900) & (T_m <= 1200)
        c_a[idx] = 650

        if np.any(T_m > 1200):
            print('WARNING - Member temperature more than 1200 degC. Outside definitions for steel heat capacity')
        idx = T_m > 1200
        c_a[idx] = 650

        return c_a

    @staticmethod
    def _calc_steel_dens():
        """Calculates steel density to BS EN 1993-1-2 Caluse 3.2.2(1) in kg/m3"""
        return 9850



