import configs as cfg
import heating_regimes as hr
import monte_carlo as mc
import pandas as pd
import json
import numpy as np
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textwrap import dedent
import collections
import itertools

# UNIT_CATALOGUE = {
#     'A_c': {'ui_label': 'ca', 'title': 'Compartment area', 'unit': 'm$^2$'},
#     'c_ratio': {'ui_label': 'csr', 'title': 'Compartment sides ratio', 'unit': '-'},
#     'h_c': {'ui_label': 'ch', 'title': 'Compartment height', 'unit': 'm'},
#     'w_frac': {'ui_label': 'vpr', 'title': 'Ventilated perimeter fraction', 'unit': '-'},
#     'h_w_eq': {'ui_label': 'heq', 'title': 'Average window height', 'unit': 'm'},
#     'remain_frac': {'ui_label': 'trf', 'title': 'Thermal resilience', 'unit': '-'},
#     'fabr_inrt': {'ui_label': 'ftr', 'title': 'Fabric thermal inertia', 'unit': 'J/m$^2$s$^{1/2}$K'},
#     'q_f_d': {'ui_label': 'fl', 'title': 'Fuel load', 'unit': 'MJ/m$^2$'},
#     'Q': {'ui_label': 'fl', 'title': 'Heat release rate per unit area', 'unit': 'KW/m$^2$'},
#     't_lim': {'ui_label': 'fl', 'title': 'Fire growth rate', 'unit': 'min'},
#     'spr_rate': {'ui_label': 'fl', 'title': 'Fire spread rate', 'unit': 'mm/s'},
#     'flap_angle': {'ui_label': 'fl', 'title': 'Flapping angle', 'unit': 'deg'},
#     'T_nf_max': {'ui_label': 'fl', 'title': 'Near field max temperature', 'unit': '°C'}}
#
# HEATING_REGIMES = {
#         'Uniform BS EN 1991-1-2': hr.UniEC1,
#         'Traveling ISO 16733-2': hr.TravelingISO16733}
# EQV_METHODS = {}
# EQV_CURVES = {}
# RISK_METHODS = {}

class AssessmentCase:

    UNITS = cfg.UNIT_CATALOGUE
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
        self.outputs = {'reliability_curve': [],  # fraction of fires less than treshold factor
                        'reliability_conf': [],  # confidence interval on reliability curve
                        'thermal_response': [],  # vector of maximum element temperatures
                        'eqv_req': None,  # vector of maximum steel temperatures
                        'eqv_req_conf': [],  # 95% confidence interval on eqv_req
                        'eqv_conc_cover': -1,   # Eqv. concrete cover
                        'success_conv': None,  # Success flag whe
                        'max_el_resp': [], # Vector of maximum element temperatures at target resistance
                        'fire_eqv': []}  # Vector of eqv. fire severtity for each fire

        self._setup_save_folder_structure(save_loc)


    def _setup_save_folder_structure(self, save_loc):
        """Setup folder structure based on analysis requirements.
        Updates case root folder"""

        self.save_loc = os.path.join(save_loc, f'{self.ID}_{self.name}')
        subfolders = ['data', 'inputs']
        for subf in subfolders:
            os.makedirs(os.path.join(self.save_loc, subf), exist_ok=True)

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

    def _estimate_eqv_req(self):
        """Interpolates convergence results to improve accuracy"""

        self.outputs['reliability_curve'] = np.array(self.outputs['reliability_curve'])
        self.outputs['reliability_curve'] = self.outputs['reliability_curve'][self.outputs['reliability_curve'][:, 0].argsort()]
        self.rel_interp_f = interpolate.interp1d(self.outputs['reliability_curve'][:, 1], self.outputs['reliability_curve'][:, 0])
        self.outputs['eqv_req'] = self.rel_interp_f(self.lim_factor)

    def _sample_sensitivity_quick(self):
        boot_res = []
        for k in range(self.configs['bootstrap_reps']):
            boot = np.random.choice(np.hstack(self.outputs['thermal_response']), len(self.outputs['thermal_response']), replace=True)
            boot_res.append(np.percentile(boot, 100*self.risk_model.risk_target))
        boot_res = self.rel_interp_f(boot_res)
        self.outputs['eqv_req_conf'] = np.percentile(boot_res, [2.5, 97.5])

    def _sample_sensitivity_full(self):
        boot_res = np.zeros((self.configs['bootstrap_reps'], len(self._eqv_assess_range)))
        for k in range(self.configs['bootstrap_reps']):
            rnd_ind = np.random.choice(range(self.outputs['thermal_response'].shape[1]), self.outputs['thermal_response'].shape[1], replace=True)
            new_resp = self.outputs['thermal_response'][:, rnd_ind]
            boot_res[k, :] = np.sum(new_resp < self.lim_factor, axis=1)/new_resp.shape[1]
        self.outputs['reliability_conf'] = np.percentile(boot_res, [2.5, 97.5], axis=0).T

        #calc confidence at target by interpolation
        self.outputs['eqv_req_conf'] = [0, 0]
        for i in range(2):
            f = interpolate.interp1d(self.outputs['reliability_conf'][:,i], self._eqv_assess_range)
            self.outputs['eqv_req_conf'][i] = f(self.risk_model.risk_target)

    def _estimate_max_elem_response_at_eqv_req(self):
        if self.analysis_type == 'full':
            f = interpolate.interp1d(self._eqv_assess_range, self.outputs['thermal_response'], axis=0)
            self.outputs['max_el_resp'] = f(self.outputs['eqv_req'])
        elif self.analysis_type == 'quick':
            self.outputs['max_el_resp'] = self.outputs['thermal_response']

    def _estimate_eqv_rating_of_all_fires(self):
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

            # Skip analysis if there are not design fires associated with this methodology
            if regime.is_empty: continue

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
        target_temp = np.percentile(thermal_response, 100 * self.risk_model.risk_target)
        reliability = np.sum(thermal_response < self.lim_factor)/len(thermal_response) #TODO Check this formula
        self.outputs['reliability_curve'].append([equiv_exp, target_temp, reliability])

        if for_optimisation:
            optm_fxn = np.sqrt((self.lim_factor - target_temp) ** 2)  # used only for optimisation
            print(f'Equiv: {equiv_exp}, Target temp: {target_temp}')
            if optm_fxn < self.max_optm_fxn[0]:
                self.max_optm_fxn[0] = optm_fxn
                try:
                    self.outputs['thermal_response'][:, 0] = thermal_response
                except TypeError:
                    self.outputs['thermal_response'] = np.vstack(thermal_response)
            return optm_fxn

        else:
            self.outputs['thermal_response'].append(thermal_response)
            self.outputs['t_hist'] = thermal_hist
            print(f'Equiv: {equiv_exp}, Reliability: {reliability}')

    def _optimise_to_limiting_factor(self):
        self.max_optm_fxn = [10000]  # holder for a value of optimisation function
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
        self.outputs['thermal_response'] = np.array(self.outputs['thermal_response'])

    def _save_design_fires_data(self, debug_return=False):
        """Processes and saves design fire database"""

        data = len(self.heating_regimes)*[0]
        begin = 0
        for i, regime in enumerate(self.heating_regimes):
            if regime.is_empty:
                continue # skip empty methodologies
            data[i] = regime.summarise_parameters(param_list='concise')
            print(begin, len(data[i]))
            data[i]['max_el_resp'] = self.outputs['max_el_resp'][begin:begin+len(data[i])]
            try:
                data[i]['fire_eqv'] = self.outputs['fire_eqv'][begin:begin + len(data[i])]
            except ValueError: # fire eqv is not available for quick analysis
                pass
            data[i] = data[i].round(3)
            begin = len(data[i])

            if debug_return:
                return data
            else:
                data[i].to_csv(os.path.join(self.save_loc, 'data', f'{self.ID}_FRS_{regime.SAVE_NAME}.csv'),
                               index_label='ID')

    def _save_thermal_response(self):
        """Saves thermal response vector as binary .npy"""
        np.save(file=os.path.join(self.save_loc, 'data', f'{self.ID}_thermal_response.npy'),
                arr=self.outputs['thermal_response'].astype('float16', copy=False))

    def _save_reliability_curve(self):
        """Saves reliability curve data as a csv"""

        data = pd.DataFrame(self.outputs['reliability_curve'], columns=['fire_severity', 'el_temp_target', 'ecdf'])
        try:
            data[['ecdf_low', 'ecdf_high']] = self.outputs['reliability_conf']
        except ValueError: # pass if sensitivity on the values is not available
            pass
        data.round(3).to_csv(os.path.join(self.save_loc, 'data', f'{self.ID}_reliability_curve.csv'), index=False)

    def _save_case_results_summary(self):

        if self.analysis_type == 'full':
            txt = f"""\
                    Full analysis for case {self.ID}-{self.name} completed successfully.\n
                    Equivalent fire severity to {self.ht_model.ecr.label}: {self.outputs['eqv_req']:.0f} min.
                    Confidence interval for sample size of {self.sample_size}: {self.outputs['eqv_req_conf'][1]:.0f} to {self.outputs['eqv_req_conf'][0]:.0f} min.
                    Required minimum depth  of concrete cover: {self.outputs['eqv_conc_cover']:.0f} mm.\n
                    Total reliability: {100*self.risk_model.total_reliability:.2f} %.
                    Sprinkler reliability:  {self.risk_model.sprinkler_reliability:.2f} %.
                    Structural reliability: {100*self.risk_model.struct_reliability:.2f} %."""
        elif self.analysis_type == 'quick':
            txt = f"""\
                    Quick analysis for case {self.ID}-{self.name} converged {'successfully' if self.outputs['success_conv'] else 'unsuccessfully'}.\n
                    Undertaken iterations: {self.optm_result.nfev}
                    Convergence error: {self.optm_result.fun:.2f}.\n
                    Equivalent fire severity to {self.ht_model.ecr.label}: {self.outputs['eqv_req']:.0f} min.
                    Confidence interval for sample size of {self.sample_size}: {self.outputs['eqv_req_conf'][1]:.0f} to {self.outputs['eqv_req_conf'][0]:.0f} min.
                    Required minimum depth  of concrete cover: {self.outputs['eqv_conc_cover']:.0f} mm.\n
                    Total reliability: {100*self.risk_model.total_reliability:.2f} %.
                    Sprinkler reliability:  {self.risk_model.sprinkler_reliability:.2f} %.
                    Structural reliability: {100*self.risk_model.struct_reliability:.2f} %."""

        with open(os.path.join(self.save_loc, f'{self.ID}_summary.txt'), 'w') as f:
            f.write(dedent(txt))

    def _plot_reliability_curve(self):
        #TODO get appropriate figure size
        sns.set()
        fig, ax = plt.subplots(figsize=(10, 6))
        # calculate hist scale factor for legibility
        binned = list(np.histogram(self.outputs['fire_eqv'],
                                   bins=int(self.configs['eqv_max'] / self.configs['eqv_step']),
                                   range=[0, self.configs['eqv_max']]))
        factor = 0.5 / np.max(binned[0] / self.sample_size)

        begin = 0
        for i, regime in enumerate(self.heating_regimes):

            if regime.is_empty: continue  # skip empty methodologies

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
        ax.hlines(y=self.risk_model.risk_target,
                  xmin=0,
                  xmax=300,
                  color='red',
                  linestyle='dashed',
                  label='Reliability target')

        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, self.outputs['reliability_curve'][self.outputs['reliability_curve'][:, 2] < 0.996][-1, 0]])
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 10))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('Equivalent fire severity rating (min)')
        ax.set_ylabel('Structural reliability')
        ax.legend()

        fig.tight_layout()
        plt.savefig(os.path.join(self.save_loc, f'{self.ID}_reliability_curve.png'),
                    bbox_inches="tight",
                    dpi=150)
        plt.close(fig)

    def _plot_convergence_study(self):
        """Plots a graph of the convergence study depicting a crude reliability curve"""

        sns.set()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.outputs['reliability_curve'][:, 0], self.outputs['reliability_curve'][:, 2], 'o',
                label='Iterations',
                color='orange')
        ax.plot(self.outputs['reliability_curve'][:, 0], self.outputs['reliability_curve'][:, 2],
                color='blue',
                linestyle='dashed',
                linewidth=1,
                label='Reliability ECDF interpolation',
                alpha=0.7)
        ax.hlines(self.risk_model.risk_target,
                  xmin=0,
                  xmax=300,
                  color='red',
                  linestyle='dashed',
                  label='Reliability target')
        ax.plot([self.outputs['eqv_req'], self.outputs['eqv_req']], [0, self.risk_model.risk_target],
                color='green',
                label='Eqv. severity requirement')
        ax.fill_betweenx(
            y=[0, self.risk_model.risk_target],
            x1=[self.outputs['eqv_req_conf'][0], self.outputs['eqv_req_conf'][0]],
            x2=[self.outputs['eqv_req_conf'][1], self.outputs['eqv_req_conf'][1]],
            alpha=0.2,
            color='green',
            label='95% conf. interval')

        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, self.configs['eqv_max']])
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 10))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('Equivalent fire severity rating (min)')
        ax.set_ylabel('Structural reliability')
        ax.legend()
        plt.savefig(os.path.join(self.save_loc, f'{self.ID}_convergence_plot.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)


    def _plot_max_el_temp_duration(self):

        mrk_lst = ['o', '^', 's', '+']
        N_max = 500  # maximum points for legibility
        vmin = 0
        vmax = np.percentile(self.outputs['fire_eqv'], 97.5)  # autoscale the colormap to the first 95%

        sns.set()
        fig, ax = plt.subplots(figsize=(10, 6))

        begin = 0
        for i, regime in enumerate(self.heating_regimes):

            if regime.is_empty: continue  # skip empty methodologies

            red_sample = int(len(regime.params['burnout']) / self.sample_size * N_max)
            sampl_ind = np.random.choice(range(len(regime.params['burnout'])), red_sample, replace=False)
            x = regime.params['burnout']
            y = self.outputs['max_el_resp'][begin:begin + len(x)]
            col = self.outputs['fire_eqv'][begin:begin + len(x)]
            _ = ax.scatter(x[sampl_ind], y[sampl_ind], c=col[sampl_ind], marker=mrk_lst[i], cmap='coolwarm',
                           vmin=vmin, vmax=vmax, label=regime.NAME, alpha=0.7)
            begin = len(x)

        ax.hlines(y=self.lim_factor, xmin=0, xmax=ax.get_xlim()[1], color='red', linestyle='dashed', label='Limiting temperature')
        ax.legend()
        ax.set_ylim([0, ax.get_ylim()[1] + 200])
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_xlabel('Fire duration (min)')
        ax.set_ylabel('Element maximum temperature (°C)')
        ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 100))
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 100))
        cb = fig.colorbar(_, extend='max')
        cb.set_label('Fire severity (min)')
        plt.savefig(os.path.join(self.save_loc, f'{self.ID}_duration_response.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_inputs(self, list_of_inputs, filename):

        sns.set()
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.ravel()

        for i, input in enumerate(list_of_inputs):
            axs[i].hist(self.inputs['values'][input], bins=25, density=True, align='mid', label='sampled data',
                               alpha=0.7)
            axs[i].plot(self.inputs['curves'][input][:, 0], self.inputs['curves'][input][:, 1],
                               linestyle='dashed', color='#FF7F0E', label='analytical target')
            axs[i].set_ylim([-axs[i].get_ylim()[1] / 100, axs[i].get_ylim()[1]])
            axs[i].set_xlabel(f'{AssessmentCase.UNITS[input]["title"]} ({AssessmentCase.UNITS[input]["unit"]})')
        [axs[k].axis('off') for k in range(i + 1, len(axs))]
        handles, labels = axs[0].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower center', ncol=2)
        plt.savefig(os.path.join(self.save_loc, 'inputs', f'{self.ID}_{filename}.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _quick_analysis(self):
        """ Analysis sequence for quick analysis"""
        self._setup_analysis_parameters()
        self.optm_result = self._optimise_to_limiting_factor()
        self._assess_convergence_success()
        self._estimate_eqv_req()
        self._sample_sensitivity_quick()
        self._estimate_max_elem_response_at_eqv_req()

        self._plot_inputs(
            list_of_inputs=['A_c', 'c_ratio', 'h_c', 'w_frac', 'h_w_eq', 'remain_frac', 'fabr_inrt'],
            filename='geometry_params')
        self._plot_inputs(
            list_of_inputs=['q_f_d', 'Q', 't_lim', 'spr_rate', 'flap_angle', 'T_nf_max'],
            filename='fire_params')
        self._plot_convergence_study()

        self._save_design_fires_data()
        self._save_reliability_curve()
        self._save_case_results_summary()
        self._save_thermal_response()

    def _full_analysis(self):
        """ Analysis sequence for full analysis"""
        self._setup_analysis_parameters()
        self._assess_full_eqv_range()
        self._estimate_eqv_req()
        self._sample_sensitivity_full()
        self._estimate_max_elem_response_at_eqv_req()
        self._estimate_eqv_rating_of_all_fires()
        self.risk_model.risk_sensitivity_study(analysis_case=self)

        self._plot_inputs(
            list_of_inputs=['A_c', 'c_ratio', 'h_c', 'w_frac', 'h_w_eq', 'remain_frac', 'fabr_inrt'],
            filename='geometry_params')
        self._plot_inputs(
            list_of_inputs=['q_f_d', 'Q', 't_lim', 'spr_rate', 'flap_angle', 'T_nf_max'],
            filename='fire_params')
        self._plot_reliability_curve()
        self._plot_max_el_temp_duration()

        self._save_case_results_summary()
        self._save_reliability_curve()
        self._save_design_fires_data()
        self._save_thermal_response()  # make sure this is always called last

    def run_analysis(self):
        """Starts analysis"""
        if self.analysis_type is 'quick':
            self._quick_analysis()
        elif self.analysis_type is 'full':
            self._full_analysis()

    def report_to_main(self):
        """Reports data to main for the purposes of cross case analysis"""
        pass

class CaseControler:

    def __init__(self, inputs, out_f):
        self.risk_method = None
        self.eqv_method = None
        self.mc_method = None
        self.cases = {}

        self.out_f = out_f
        self.inputs = inputs
        self._setup_folder_structure()
        self._update_with_sys_configs()
        self._save_inputs()

    def _update_with_sys_configs(self):
        """Updates the input file with non-user controlled configurations"""

        for method in ['eqv_method', 'eqv_curve', 'heating_regimes', 'risk_method']:

            #if input is not dict of dict then transform it as dictionary
            if not isinstance(self.inputs[method], dict):
                self.inputs[method] = {self.inputs[method]: {}}
            for i in self.inputs[method]:
                self.inputs[method][i].update(cfg.METHODOLOGIES[method][i][1])

    def _save_inputs(self):
        with open(os.path.join(self.out_f, 'info', 'inputs.json'), 'w') as f:
            json.dump(self.inputs, f, indent=4)

    def _setup_folder_structure(self):
        subfolders = ['info', 'run_a', 'run_b', 'eqv']
        for subf in subfolders:
            os.makedirs(os.path.join(self.out_f, subf), exist_ok=True)

    def initiate_methods(self):
        """Method performs number of initial calculations before main analysis"""

        # Setup risk method
        m_label = list(self.inputs['risk_method'].keys())[0]
        m_config = self.inputs['risk_method'][m_label]
        self.risk_method = cfg.METHODOLOGIES['risk_method'][m_label][0](**m_config)
        # Setup eqv method
        m_label = list(self.inputs['eqv_method'].keys())[0]
        eqv_curve = list(self.inputs['eqv_curve'].keys())[0]
        m_config = self.inputs['eqv_method'][m_label]
        self.eqv_method = cfg.METHODOLOGIES['eqv_method'][m_label][0](
            equivalent_curve=eqv_curve, **m_config)
        # Setup mc method
        self.mc_method = mc.ProbControl(seed=self.inputs['random_seed'])

        # setup run a cases

    def conduct_pre_run_calculations(self):
        # calculate risk target
        self.risk_method.assess_risk_target()
        # calculate eqv_protection
        self.eqv_method.get_equivalent_protection()
        #TODO plot equivalent protection

    def _get_cases(self):

        # Get only inputs having more than one name

        # make list of name parameters based on iteration number.
        # reject those with only one parameters
        # concatinate into a name

        #Order the dictionary of parameters to ensure correct indexing
        self.inputs['parameters'] = collections.OrderedDict(sorted(self.inputs['parameters'].items()))

        if self.inputs['param_mode'] == 'grid':

            tmp_list_inputs = [self.inputs['parameters'][k] for k in self.inputs['parameters']]
            tmp_list_numeric = [list(range(1, len(k)+1)) for k in tmp_list_inputs]
            all_cases = itertools.product(*tmp_list_inputs)
            all_cases_numeric = itertools.product(*tmp_list_numeric)
            for i, (case, num) in enumerate(zip(all_cases, all_cases_numeric)):
                self.cases[f'{(i+1):03d}'] = {
                    'label': CaseControler._create_case_name(params=self.inputs['parameters'],  num_mask=num, num_map=tmp_list_numeric),
                    'params': {label: case for (label, case) in zip(self.inputs['parameters'], case)}}
        elif self.parametrisation == 'one_at_a_time':
            pass
        elif self.parametrisation == 'parallel':
            pass

    @staticmethod
    def _create_case_name(params, num_mask, num_map):
        # TODO - write documentation and refractor to be applicable for all cases
        tmp = [f'{cfg.UNIT_CATALOGUE[key]["ui_label"]}{j}' for (key, j, m) in zip(params, num_mask, num_map) if len(m) != 1]
        return '_'.join(tmp)


    def run_a_study(self):
        pass

    def process_a_study_results(self):
        pass

    def run_b_study(self):
        pass
