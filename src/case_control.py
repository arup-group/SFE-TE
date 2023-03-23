import heating_regimes as hr
import numpy as np
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

class AssessmentCase:
    HEATING_REGIMES = {
        'Uniform BS EN 1991-1-2': hr.UniEC1,
        'Traveling ISO 16733-2': hr.TravelingISO16733}

    def __init__(self, name, ID, input_defs, risk_model, mc_engine, ht_model, heating_regimes_inputs,
                 lim_factor, save_loc, analysis_type, sample_size, configs):
        self.name = name
        self.ID = ID
        self.input_defs = input_defs
        self.risk_model = risk_model
        self.mc_engine = mc_engine  # class instance to the
        self.ht_model = ht_model
        self.heating_regimes_inputs = heating_regimes_inputs  # dict list of heating regimes
        self.lim_factor = lim_factor  # limiting temperature
        self.analysis_type = analysis_type # quick of full
        self.sample_size = sample_size
        self.configs = configs  # Analysis configs #TODO Tidy up configurations

        # Variables which will be updated
        self.case_root_folder = None
        self.heating_regimes = []
        self.inputs = None
        self.optm_result = None
        self.max_optm_fxn = None
        self.rel_interp_f = None
        self._eqv_assess_range = None
        self.outputs = {'reliability_curve': [],
                        'reliability_conf': None,
                        'thermal_response': [],
                        'eqv_req': None,
                        'eqv_req_conf': None,
                        'success_conv': None,
                        'max_el_resp': None,
                        'fire_eqv': []}

        self._setup_save_folder_structure(save_loc)


    def _setup_save_folder_structure(self, save_loc):
        """Setup folder structure based on analysis requirements.
        Updates case root folder"""

        pass

    def _setup_analysis_parameters(self):
        #Draw MC parameters
        self.inputs = self.mc_engine.sample_inputs(inputs=self.input_defs, sample_size=self.sample_size)

        #initialize heating regimes
        for i, regime in enumerate(self.heating_regimes_inputs):
            self.heating_regimes.append(
                AssessmentCase.HEATING_REGIMES[regime](
                    design_fire_inputs=self.inputs['values'], **self.heating_regimes_inputs[regime]))
            self.heating_regimes[i].perform_initial_calculations()
            self.heating_regimes[i].check_bad_samples()

    def _assess_convergence_success(self):
        """Checks whether convergence is within tolerance limits"""
        self.outputs['success_conv'] = self.optm_result.fun < self.ht_model.optm_config['tol']

    def _interpolate_equiv(self):
        """Interpolates convergence results to improve accuracy"""

        self.outputs['reliability_curve'] = np.array(self.outputs['reliability_curve'])
        self.outputs['reliability_curve'] = self.outputs['reliability_curve'][self.outputs['reliability_curve'][:, 0].argsort()]
        self.rel_interp_f = interpolate.interp1d(self.outputs['reliability_curve'][:, 1], self.outputs['reliability_curve'][:, 0])
        self.outputs['eqv_req'] = self.rel_interp_f(self.lim_factor)

    def _sample_sensitivity_quick(self):
        boot_res = []
        for k in range(self.configs['bootstrap_reps']):
            boot = np.random.choice(self.outputs['thermal_response'][0], len(self.outputs['thermal_response'][0]), replace=True)
            boot_res.append(np.percentile(boot, 100*self.risk_model['target']))
        boot_res = self.rel_interp_f(boot_res)

        self.outputs['eqv_req_conf'] = np.percentile(boot_res, [2.5, 97.5])

    def _sample_sensitivity_full(self):
        boot_res = np.zeros((self.configs['bootstrap_reps'], len(self._eqv_assess_range)))
        self.outputs['thermal_response'] = np.array(self.outputs['thermal_response'])
        for k in range(self.configs['bootstrap_reps']):
            rnd_ind = np.random.choice(range(self.outputs['thermal_response'].shape[1]), self.outputs['thermal_response'].shape[1], replace=True)
            new_resp = self.outputs['thermal_response'][:, rnd_ind]
            boot_res[k, :] = np.sum(new_resp < self.lim_factor, axis=1)/new_resp.shape[1]
        self.outputs['reliability_conf'] = np.percentile(boot_res, [2.5, 97.5], axis=0).T

        #calc confidence at target by interpolation
        self.outputs['eqv_req_conf'] = [0, 0]
        for i in range(2):
            f = interpolate.interp1d(self.outputs['reliability_conf'][:,i], self._eqv_assess_range)
            self.outputs['eqv_req_conf'][i] = f(self.risk_model['target'])

    def _estimate_max_elem_response(self):
        f = interpolate.interp1d(self._eqv_assess_range, self.outputs['thermal_response'], axis=0)
        self.outputs['max_el_resp'] = f(self.outputs['eqv_req'])

    def _estimate_fire_eqv(self):
        for i in range(self.outputs['thermal_response'].shape[1]):
            f = interpolate.interp1d(
                self.outputs['thermal_response'][:, i], self._eqv_assess_range, fill_value=(self.configs['eqv_max']+30, 2), bounds_error=False)
            self.outputs['fire_eqv'].append(f(self.lim_factor))
        self.outputs['fire_eqv'] = np.array(self.outputs['fire_eqv']).round(0)

    def _assess_single_equiv(self, equiv_exp, for_optimisation=False):
        #TODO EXPLAIN DOCUMENTATION

        thermal_response = []
        thermal_hist = []
        for regime in self.heating_regimes:
            T_max, T_hist = self.ht_model.calc_thermal_response(
                equiv_exp=equiv_exp,
                exposure_fxn=regime.get_exposure,
                t_final=np.max(regime.params['burnout']),
                sample_size=len(regime.params['burnout']),
                output_history=False,  # set to default. Use True only for debugging
                early_stop=20)
            thermal_response.append(T_max)
            thermal_hist.append(T_hist)

        thermal_response = np.concatenate(thermal_response)
        target_temp = np.percentile(thermal_response, 100 * self.risk_model['target'])
        reliability = np.sum(thermal_response < self.lim_factor)/len(thermal_response) #TODO Check this formula
        self.outputs['reliability_curve'].append([equiv_exp, target_temp, reliability])

        if for_optimisation:
            optm_fxn = np.sqrt((self.lim_factor - target_temp) ** 2)  # used only for optimisation
            print(f'Equiv: {equiv_exp}, Target temp: {target_temp}')
            if optm_fxn < self.max_optm_fxn[0]:
                self.max_optm_fxn[0] = optm_fxn
                try:
                    self.outputs['thermal_response'][0] = thermal_response
                except IndexError:
                    self.outputs['thermal_response'].append(thermal_response)
            return optm_fxn
        else:
            self.outputs['thermal_response'].append(thermal_response)
            self.outputs['t_hist'] = thermal_hist
            print(f'Equiv: {equiv_exp}, Reliability: {reliability}')

    def _optimise_to_limiting_factor(self):
        return optimize.minimize_scalar(
            lambda x: self._assess_single_equiv(x, for_optimisation=True),
            bounds=(1, self.ht_model.eqv_max),
            method='bounded',
            options={'maxiter': self.ht_model.optm_config['max_itr'],
                     'xatol': self.ht_model.optm_config['tol']})

    def _assess_full_eqv_range(self):

        self._eqv_assess_range = np.arange(5, self.configs['eqv_max'], self.configs['eqv_step'])
        self._eqv_assess_range = np.append(self._eqv_assess_range, self.configs['eqv_max'])
        for t_eqv in self._eqv_assess_range:
            self._assess_single_equiv(t_eqv, for_optimisation=False)

    def _quick_analysis(self):
        """ Analysis sequence for quick analysis"""
        self._setup_analysis_parameters()
        self.max_optm_fxn = [10000]
        self.optm_result = self._optimise_to_limiting_factor()
        self._assess_convergence_success()
        self._interpolate_equiv()
        self._sample_sensitivity_quick()

    def _full_analysis(self):
        """ Analysis sequence for full analysis"""
        self._setup_analysis_parameters()
        self._assess_full_eqv_range()
        self._interpolate_equiv()
        self._sample_sensitivity_full()
        self._estimate_max_elem_response()
        self._estimate_fire_eqv()

    def run_analysis(self):
        """Starts analysis"""
        if self.analysis_type is 'quick':
            self._quick_analysis()
        elif self.analysis_type is 'full':
            self._full_analysis()

    def _save_design_fires_data(self, debug_return):
        """Processes and saves design fire database"""

        data = len(self.heating_regimes)*[0]
        begin = 0
        for i, regime in enumerate(self.heating_regimes):
            data[i] = regime.summarise_parameters(param_list='concise')
            data[i]['max_el_resp'] = self.outputs['max_el_resp'][begin:begin+len(data[i])]
            data[i]['fire_eqv'] = self.outputs['fire_eqv'][begin:begin + len(data[i])]
            data[i] = data[i].round(3)
            begin = len(data[i])
            #TODO add saving statement once confirm folder structure

        if debug_return:
            return data

    def plot_reliability_curve(self, debug_show):

        sns.set()
        fig, ax = plt.subplots()

        # calculate hist scale factor for legibility
        binned = list(np.histogram(self.outputs['fire_eqv'],
                                   bins=int(self.configs['eqv_max'] / self.configs['eqv_step']),
                                   range=[0, self.configs['eqv_max']]))
        factor = 0.5 / np.max(binned[0] / self.sample_size)

        begin = 0
        for i, regime in enumerate(self.heating_regimes):
            data = self.outputs['fire_eqv'][begin:begin + len(regime.params['A_c'])]
            binned = list(np.histogram(data, bins=int(self.configs['eqv_max'] / self.configs['eqv_step']),
                                       range=[0, self.configs['eqv_max']]))
            binned[0] = np.array(binned[0]) / self.sample_size * factor
            ax.bar(x=binned[1][:-1], height=binned[0], width=np.diff(binned[1]), align='edge', alpha=0.5,
                   label=regime.NAME)
            begin = len(regime.params['A_c'])
        ax.plot(self.outputs['reliability_curve'][:, 0], self.outputs['reliability_curve'][:, 2],
                color='black',
                label='Reliability ECDF')
        ax.plot(self.outputs['reliability_curve'][:, 0], self.outputs['reliability_conf'][:, 0],
                color='grey',
                linestyle='dashed',
                linewidth=1,
                label='95% conf. interval')
        ax.plot(self.outputs['reliability_curve'][:, 0], self.outputs['reliability_conf'][:, 1],
                color='grey',
                linestyle='dashed',
                linewidth=1)
        ax.hlines(y=self.risk_model['target'],
                  xmin=0,
                  xmax=300,
                  color='red',
                  linestyle='dashed',
                  label='Reliability target')

        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, self.outputs['reliability_curve'][self.outputs['reliability_curve'][:, 2] < 0.996][-1, 0]])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('Equivalent fire severity rating (min)')
        ax.set_ylabel('Structural reliability')
        ax.legend(prop={'size': 9})

        #TODO Save figure
        if debug_show:
            fig.show()


    def report_to_main(self):
        """Reports data to main for the purposes of cross case analysis"""
        pass

    def save_data(self):
        """Saves data in case folder"""
        pass

    def process_plots(self):
        """Processes and saves relevant plots"""



        pass