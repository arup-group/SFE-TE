import heating_regimes as hr
import numpy as np
from scipy import optimize

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
        self.configs = configs  # Analysis configs

        # Variables which will be updated
        self.case_root_folder = None
        self.heating_regimes = []
        self.inputs = None
        self.optm_result = None
        self.max_optm_fxn = None
        self.outputs = {'reliability_curve': [],
                        'thermal_response': [],
                        'equiv': None}

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

    def _compute_equivalence(self):
        pass

    def _assess_sample_sensitivity(self):
        pass

    def run_analysis(self):
        """Starts analysis"""
        self._setup_analysis_parameters()

        if self.analysis_type is 'quick':
            self.max_optm_fxn = [10000]
            self.optm_result = self._optimise_to_limiting_factor()
        elif self.analysis_type is 'complete':
            pass

        # Assess sensitivity
        # Assess individual equivalence




    def _assess_single_equiv(self, equiv_exp, for_optimisation = False):
        #TODO EXPLAIN DOCUMENTATION

        thermal_response = []
        for regime in self.heating_regimes:
            T_max, T_hist = self.ht_model.calc_thermal_response(
                equiv_exp=equiv_exp,
                exposure_fxn=regime.get_exposure,
                t_final=np.max(regime.params['burnout']),
                sample_size=len(regime.params['burnout']),
                output_history=False,  # set to default. Use True only for debugging
                early_stop=15)
            thermal_response.append(T_max)

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

    def _optimise_to_limiting_factor(self):
        return optimize.minimize_scalar(
            lambda x: self._assess_single_equiv(x, for_optimisation=True),
            bounds=(1, self.ht_model.eqv_max),
            method='bounded',
            options={'maxiter': self.ht_model.optm_config['max_itr'],
                     'xatol': self.ht_model.optm_config['tol']})

    def report_to_main(self):
        """Reports data to main for the purposes of cross case analysis"""
        pass

    def save_data(self):
        """Saves data in case folder"""
        pass

    def process_plots(self):
        """Processes and saves relevant plots"""
        pass