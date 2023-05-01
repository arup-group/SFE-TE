import configs as cfg
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class GenericHT():
    """Generic class for heat transfer analysis"""
    LABEL = 'Generic'
    DESCR = 'Generic description'

    def __init__(self, equivalent_curve):
        self.ecr = self._load_equivalent_curve(equivalent_curve)
        self.equiv_prot_req_data = None
        self.equiv_prot_interp = None
        self.eqv_step = None  # Incremental step at which equivalency is assessed
        self.eqv_max = None  # Maximum equivalence range
        self.lim_factor = None  # Parameter value at which eqv. is asssessed
        self.max_itr = None  # Max iterations for convergence study
        self.tol = None  # Max tolerance for convergence study

    def _load_equivalent_curve(self, equivalent_curve):
        return cfg.METHODOLOGIES['eqv_curve'][equivalent_curve][0]()

    def _process_sample_section_geometry(self, sect_prop):
        raise NotImplementedError

    def get_equivalent_protection(self):
        raise NotImplementedError

    def plot_equivelant_protection_curve(self):
        raise NotImplementedError

    def calc_thermal_response(self):
        raise NotImplementedError


class SteelEC3(GenericHT):
    # Some constants
    LABEL = 'Steel EC3 HT'
    DESCR = '0D heat transfer in accordance with BS EN 1993-1-2'

    def __init__(self, equivalent_curve, A_v, c_p, k_p, ro_p, lim_temp, dt, T_amb,
                 max_itr, tol, eqv_max, eqv_step, prot_thick_range):
        super().__init__(equivalent_curve)
        self.prot_prop = {'c_p': c_p, 'k_p': k_p, 'ro_p': ro_p}  # TODO to be refactored
        self.sect_prop = self._process_sample_section_geometry({'A_v': A_v})  # TODO to be refactored
        self.T_lim = lim_temp
        self.T_amb = T_amb
        self.dt = dt  # time step size
        self.max_itr = max_itr
        self.tol = tol
        self.eqv_max = eqv_max
        self.eqv_step = eqv_step
        self.prot_thick_range = prot_thick_range

        self._issue_steel_hc_warn = [True, True]  # Counter for issuing warning from steel hc only once
        self.limiting_factor = self.T_lim

        # TODO create interpolate prot_thickness wrapper to give warnings when extrapolating
        # TODO create protection thickness plot and graph

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

        if 'A_v' in sect_prop:
            return {'A_v': sect_prop['A_v']}

        # calculate area
        A = 2 * sect_prop['fl_t'] * sect_prop['B'] + (sect_prop['D'] - 2 * sect_prop['fl_t']) * sect_prop['wb_t']
        # calculate heated perimeter for either 4 or 3 side exposure
        if sect_prop['exp_sides'] == 'four':
            P = 2 * sect_prop['D'] + 4 * sect_prop['B'] - 2 * sect_prop['wb_t']
        elif sect_prop['exp_sides'] == 'three':
            P = 2 * sect_prop['D'] + 3 * sect_prop['B'] - 2 * sect_prop['wb_t']
        return {'A_v': P / A}

    def get_equivalent_protection(self):
        """Calculates an interpolation curve of required fire protection material for equivalent protection
        thickness based on given gas temperature curve, section, material, and protection material properties.
         INTEGRATION TEST REQUIRED - to be checked against predictions of the SFE toolkit
         """

        # Get some properties
        c_p = self.prot_prop['c_p']
        k_p = self.prot_prop['k_p']
        ro_p = self.prot_prop['ro_p']
        A_v = self.sect_prop['A_v']

        # Get initial condition for eqv. curve method
        ini_temp = self.ecr.get_initial_condition()

        # Get array of protection thicknesses
        prot_thick = np.arange(self.prot_thick_range[0], self.prot_thick_range[1], self.prot_thick_range[2])

        # Create initial temperature array equal to ambient of same shape
        T_m = np.full_like(prot_thick, ini_temp)

        times = np.arange(0, 60*self.eqv_max, self.dt)

        # Get a holder for intermediate results for debugging
        all_temps = np.full((len(times), len(prot_thick)), -1, dtype=np.float64)
        all_dTs = np.full((len(times), len(prot_thick)), -1, dtype=np.float64)
        all_Tgas = np.full((len(times), 1), -1, dtype=np.float64)
        all_fi = np.full((len(times), len(prot_thick)), -1, dtype=np.float64)
        all_ca = np.full((len(times), len(prot_thick)), -1, dtype=np.float64)

        # Start Forward Euler solution of thermal response - see BS EN 1993-1-2 Clause 4.2.5.2(1)
        for i, t in enumerate(times):
            all_temps[i, :] = T_m

            c_a = self._calc_steel_hc(T_m)
            ro_a = SteelEC3._calc_steel_dens()

            T_g = self.ecr.get_temp(t+ self.dt)
            T_g_prev= self.ecr.get_temp(t)

            fi = c_p * ro_p * prot_thick * A_v / (c_a * ro_a)

            dT = k_p * A_v * (T_g - T_m) * self.dt / ((prot_thick * c_a * ro_a) * (1 + fi / 3)) - (np.exp(fi/10)-1)*(T_g - T_g_prev)
            dT[(dT < 0) & (T_g - T_g_prev > 0)] = 0  # enforcing condition for BS EN 1993-1-2 eq. 4.27
            T_m = T_m + dT
            T_m[T_m < ini_temp] = ini_temp  # Temperature cannot go below eqv. curve initial condition

            #kept for debugging purposes - TO BE REMOVED ONCE COMPLETE
            all_fi[i, :] = fi
            all_ca[i, :] = c_a
            all_dTs[i, :] = dT
            all_Tgas[i, :] = T_g

        debug_results = {
            'times': times,
            'all_temps': all_temps,
            'all_dTs': all_dTs,
            'all_Tgas': all_Tgas,
            'all_fi': all_fi,
            'all_ca': all_ca}

        # Start interpolation for protection thickness
        self.equiv_prot_req_data = np.zeros((len(prot_thick), 2))
        self.equiv_prot_req_data[:, 0] = prot_thick

        for i in range(all_temps.shape[1]):
            f = interp1d(all_temps[:, i], times)
            try:
                self.equiv_prot_req_data[i, 1] = f(self.T_lim)/60
            except ValueError: # in case limiting temperature is not reached
                self.equiv_prot_req_data[i, 1] = -1

        # remove rows where limiting temperature was not reached
        self.equiv_prot_req_data = self.equiv_prot_req_data[self.equiv_prot_req_data[:, 1] != -1]

        # create interpolation object
        self.equiv_prot_interp = interp1d(
            x=self.equiv_prot_req_data[:, 1],
            y=self.equiv_prot_req_data[:, 0],
            fill_value='extrapolate')

        return debug_results

    def report_eqv_data(self, save_loc):
        """Plots equivalent protection curve and saves a csv of supporting data
        Inputs:
            where (str): save location
        """
        data = pd.DataFrame(self.equiv_prot_req_data, columns=['Prot. thickness', f'{self.ecr.label} exposure'])
        data['Prot. thickness'] = 1000*data['Prot. thickness']
        sns.set()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data[f'{self.ecr.label} exposure'], data['Prot. thickness'])
        ax.set_xlabel(f'Exposure to {self.ecr.label} (min)')
        ax.set_ylabel(f'Protection thickness to  {self.T_lim} °C lim. temperature (mm)')

        data.to_csv(os.path.join(save_loc, 'eqv', 'eqv_data.csv'), index=False)
        plt.savefig(os.path.join(save_loc, 'eqv', 'eqv_data.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)


    def calc_thermal_response(self, equiv_exp, exposure_fxn, t_final, sample_size, output_history, early_stop):
        """Calculates the thermal response of the sample section against an array of design fires representative
        of a single exposure regime. INTEGRATION TEST REQUIRED

        Inputs:
            exposure_fxn(method): exposure function defining the gas temperature at different points in time. It must
            be of the form f(t, *args) where t is the time. Return value to be in (degC)
            equiv_exp (float): equivalent exposure rating used to calculate appropriate protection thickness
            t_final (float): end analysis time
            sample_size (int): sample size
            early_stop (float): Difference between maximum temperature and steel member temperature at which the computation
            for this realization is stopped tp improve performance. NOTE: This works correctly only for heating regimes
            with one expected peak.

        Returns:
            max_temps (array like) - array of max temperatures,
            all_temps (array_like) - array of complete thermal response history in shape (sample size x times)"""

        # Get some properties
        c_p = self.prot_prop['c_p']
        k_p = self.prot_prop['k_p']
        ro_p = self.prot_prop['ro_p']
        A_v = self.sect_prop['A_v']

        # Get array of protection thicknesses
        d_p = self.equiv_prot_interp(equiv_exp)

        times = np.arange(0, 60*t_final, self.dt)

        # Create initial temperature array equal to ambient of same shape
        T_m = np.full(sample_size, self.T_amb, dtype=np.float64)
        T_max = np.full(sample_size, self.T_amb, dtype=np.float64)
        to_compute = np.full(sample_size, True)

        # Holder for results
        if output_history:
            all_temps = np.full((len(times), sample_size), -1, dtype=np.float64)
        else:
            all_temps = None

        for i, t in enumerate(times):
            if all_temps is not None:
                all_temps[i, to_compute] = T_m[to_compute]

            T_m_red = T_m[to_compute]

            c_a = self._calc_steel_hc(T_m_red)
            ro_a = SteelEC3._calc_steel_dens()

            #TODO ENSURE CONSISTENT TIMES
            T_g = exposure_fxn((t+self.dt)/60, subsample_mask=to_compute)
            T_g_prev= exposure_fxn(t/60, subsample_mask=to_compute)

            fi = c_p * ro_p * d_p * A_v / (c_a * ro_a)
            dT = k_p * A_v * (T_g - T_m_red) * self.dt / ((d_p * c_a * ro_a) * (1 + fi / 3)) - (np.exp(fi/10)-1)*(T_g - T_g_prev)
            dT[(dT<0) & (T_g - T_g_prev > 0)] = 0 # enforcing condition for BS EN 1993-1-2 eq. 4.27

            T_m_red = T_m_red + dT
            T_m_red[T_m_red < self.T_amb] = self.T_amb  # Temperature cannot go below ambient. TO BE CHECKED

            T_m[to_compute] = T_m_red

            to_compute = T_max - T_m < early_stop
            T_max[T_max < T_m] = T_m[T_max < T_m]

            if np.all(to_compute == False):
                break

        return T_max, all_temps

    def _calc_steel_hc(self, T_m):
        """Vectorised calculation for steel heat capacity in accordance with BS EN 1993-1-2 Clause 3.4.1.2
        UNIT TEST REQUIRED

        Inputs:
            T (flor or array like) : temperature in [degC]"""

        if not isinstance(T_m, np.ndarray):
            T_m = np.asarray(T_m, dtype=np.float64)

        c_a = np.zeros_like(T_m)
        if np.any(T_m < 20) and self._issue_steel_hc_warn[0]:
            print('WARNING - Member temperature less than 20 degC. Outside definitions for steel heat capacity')
            self._issue_steel_hc_warn[0] = False

        idx = T_m < 20
        c_a[idx] = 425 + 0.773 * T_m[idx] - 0.00169 * T_m[idx] ** 2 + 0.00000222 * T_m[idx] ** 3

        idx = (T_m >= 20) & (T_m < 600)
        c_a[idx] = 425 + 0.773 * T_m[idx] - 0.00169 * T_m[idx] ** 2 + 0.00000222 * T_m[idx] ** 3

        idx = (T_m >= 600) & (T_m < 735)
        c_a[idx] = 666 + 13002 / (738 - T_m[idx])

        idx = (T_m >= 735) & (T_m < 900)
        c_a[idx] = 545 + 17820 / (T_m[idx] - 731)

        idx = (T_m >= 900) & (T_m <= 1200)
        c_a[idx] = 650

        if np.any(T_m > 1200) and self._issue_steel_hc_warn[1]:
            print('WARNING - Member temperature more than 1200 degC. Outside definitions for steel heat capacity. Assigned value: 650 J/kgK.')
            self._issue_steel_hc_warn[1] = False

        idx = T_m > 1200
        c_a[idx] = 650

        return c_a

    @staticmethod
    def _calc_steel_dens():
        """Calculates steel density to BS EN 1993-1-2 Clause 3.2.2(1) in kg/m3"""
        return 7850
