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


class AssessmentCase:

    UNITS = cfg.UNIT_CATALOGUE
    HEATING_REGIMES = cfg.METHODOLOGIES['heating_regimes']

    def __init__(self, name, ID, input_defs, risk_model, mc_engine, ht_model, heating_regimes_inputs,
                 save_loc, analysis_type, sample_size, bootstrap_rep):
        self.name = name
        self.ID = ID
        self.input_defs = input_defs
        self.risk_model = risk_model
        self.mc_engine = mc_engine  # class instance to the
        self.ht_model = ht_model
        self.heating_regimes_inputs = heating_regimes_inputs  # dict list of heating regimes
        self.analysis_type = analysis_type # quick of full
        self.sample_size = sample_size
        self.bootstrap_rep = bootstrap_rep

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

        self.lim_factor = self.ht_model.limiting_factor
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
                AssessmentCase.HEATING_REGIMES[regime][0](
                    design_fire_inputs=self.inputs['values'], **self.heating_regimes_inputs[regime]))
            self.heating_regimes[i].perform_initial_calculations()
            self.heating_regimes[i].check_bad_samples()

    def _assess_convergence_success(self):
        """Checks whether convergence is within tolerance limits"""
        self.outputs['success_conv'] = self.optm_result.fun < self.ht_model.tol

    def _estimate_eqv_req(self):
        """Interpolates convergence results to improve accuracy"""

        self.outputs['reliability_curve'] = np.array(self.outputs['reliability_curve'])
        self.outputs['reliability_curve'] = self.outputs['reliability_curve'][self.outputs['reliability_curve'][:, 0].argsort()]
        self.rel_interp_f = interpolate.interp1d(self.outputs['reliability_curve'][:, 1], self.outputs['reliability_curve'][:, 0])
        self.outputs['eqv_req'] = self.rel_interp_f(self.lim_factor)

    def _sample_sensitivity_quick(self):
        boot_res = []
        for k in range(self.bootstrap_rep):
            boot = np.random.choice(np.hstack(self.outputs['thermal_response']), len(self.outputs['thermal_response']), replace=True)
            boot_res.append(np.percentile(boot, 100*self.risk_model.risk_target))
        boot_res = self.rel_interp_f(boot_res)
        self.outputs['eqv_req_conf'] = np.percentile(boot_res, [2.5, 97.5])

    def _sample_sensitivity_full(self):
        boot_res = np.zeros((self.bootstrap_rep, len(self._eqv_assess_range)))
        for k in range(self.bootstrap_rep):
            rnd_ind = np.random.choice(range(self.outputs['thermal_response'].shape[1]), self.outputs['thermal_response'].shape[1], replace=True)
            new_resp = self.outputs['thermal_response'][:, rnd_ind]
            boot_res[k, :] = np.sum(new_resp < self.lim_factor, axis=1)/new_resp.shape[1]
        self.outputs['reliability_conf'] = np.percentile(boot_res, [2.5, 97.5], axis=0).T

        #calc confidence at target by interpolation
        self.outputs['eqv_req_conf'] = np.array([0, 0], dtype='float')
        for i in range(2):
            f = interpolate.interp1d(self.outputs['reliability_conf'][:,i], self._eqv_assess_range)
            self.outputs['eqv_req_conf'][i] = float(f(self.risk_model.risk_target))

    def _estimate_max_elem_response_at_eqv_req(self):
        if self.analysis_type == 'full':
            f = interpolate.interp1d(self._eqv_assess_range, self.outputs['thermal_response'], axis=0)
            self.outputs['max_el_resp'] = f(self.outputs['eqv_req'])
        elif self.analysis_type == 'quick':
            self.outputs['max_el_resp'] = self.outputs['thermal_response']

    def _estimate_eqv_rating_of_all_fires(self):
        for i in range(self.outputs['thermal_response'].shape[1]):
            f = interpolate.interp1d(
                self.outputs['thermal_response'][:, i], self._eqv_assess_range, fill_value=(self.ht_model.eqv_max+30, 2), bounds_error=False)
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
            # print(f'Equiv: {equiv_exp}, Target temp: {target_temp}')
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
            options={'maxiter': self.ht_model.max_itr,
                     'xatol': self.ht_model.tol})

    def _assess_full_eqv_range(self):

        self._eqv_assess_range = np.arange(5, self.ht_model.eqv_max, self.ht_model.eqv_step)
        self._eqv_assess_range = np.append(self._eqv_assess_range, self.ht_model.eqv_max)
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
                    Converged to equivalent fire severity to {self.ht_model.ecr.label}: {self.outputs['eqv_req']:.0f} min.
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
                                   bins=int(self.ht_model.eqv_max / self.ht_model.eqv_step),
                                   range=[0, self.ht_model.eqv_max]))
        factor = 0.5 / np.max(binned[0] / self.sample_size)

        begin = 0
        for i, regime in enumerate(self.heating_regimes):

            if regime.is_empty: continue  # skip empty methodologies

            data = self.outputs['fire_eqv'][begin:begin + len(regime.params['A_c'])]
            binned = list(np.histogram(data, bins=int(self.ht_model.eqv_max / self.ht_model.eqv_step),
                                       range=[0, self.ht_model.eqv_max]))
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
        ax.set_xlim([0, self.ht_model.eqv_max])
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
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        axs = axs.ravel()

        begin = 0
        for i, regime in enumerate(self.heating_regimes):

            if regime.is_empty: continue  # skip empty methodologies

            red_sample = int(len(regime.params['burnout']) / self.sample_size * N_max)
            sampl_ind = np.random.choice(range(len(regime.params['burnout'])), red_sample, replace=False)
            x = regime.params['burnout']
            y1 = regime.params['max_gas_temp']
            y2 = self.outputs['max_el_resp'][begin:begin + len(x)]
            col = self.outputs['fire_eqv'][begin:begin + len(x)]
            axs[0].scatter(x[sampl_ind], y1[sampl_ind], c=col[sampl_ind], marker=mrk_lst[i], cmap='coolwarm',
                           vmin=vmin, vmax=vmax, label=regime.NAME, alpha=0.7)
            _ = axs[1].scatter(x[sampl_ind], y2[sampl_ind], c=col[sampl_ind], marker=mrk_lst[i], cmap='coolwarm',
                           vmin=vmin, vmax=vmax, label=regime.NAME, alpha=0.7)
            begin = len(x)

        axs[1].hlines(y=self.lim_factor, xmin=0, xmax=axs[1].get_xlim()[1], color='red', linestyle='dashed', label='Limiting temperature')
        axs[1].set_ylim([0, axs[1].get_ylim()[1] + 200])
        axs[1].set_xlim([0, axs[1].get_xlim()[1]])
        axs[1].set_xlabel('Fire duration (min)')
        axs[1].set_ylabel('Maximum element temperature (°C)')
        axs[1].set_yticks(np.arange(axs[1].get_ylim()[0], axs[1].get_ylim()[1], 100))
        axs[1].set_xticks(np.arange(axs[1].get_xlim()[0], axs[1].get_xlim()[1], 100))
        axs[1].legend()
        axs[0].set_ylim([0, axs[0].get_ylim()[1] + 200])
        axs[0].set_xlim([0, axs[0].get_xlim()[1]])
        axs[0].set_xlabel('Fire duration (min)')
        axs[0].set_ylabel('Maximum gas temperature (°C)')
        axs[0].set_yticks(np.arange(axs[0].get_ylim()[0], axs[0].get_ylim()[1], 100))
        axs[0].set_xticks(np.arange(axs[0].get_xlim()[0], axs[0].get_xlim()[1], 100))
        axs[0].legend()
        cb = fig.colorbar(_, extend='max')
        cb.set_label('Fire severity (min)')
        plt.savefig(os.path.join(self.save_loc, f'{self.ID}_duration_response.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def _plot_max_el_temp_duration_quick(self):

        mrk_lst = ['o', '^', 's', '+']
        N_max = 600  # maximum points for legibility
        vmin = np.percentile(self.outputs['max_el_resp'], 2.5)
        vmax = np.percentile(self.outputs['max_el_resp'], 97.5)  # autoscale the colormap to the first 95%

        sns.set()
        fig, ax = plt.subplots(figsize=(10, 6))

        begin = 0
        for i, regime in enumerate(self.heating_regimes):

            if regime.is_empty: continue  # skip empty methodologies

            red_sample = int(len(regime.params['burnout']) / self.sample_size * N_max)
            sampl_ind = np.random.choice(range(len(regime.params['burnout'])), red_sample, replace=False)
            x = regime.params['burnout']
            y1 = regime.params['max_gas_temp']
            col = self.outputs['max_el_resp'].reshape(-1)[begin:begin + len(x)]
            design_mask = (col < self.lim_factor + 5) & (col > self.lim_factor - 5)

            tmp_mask = np.zeros(design_mask.size, dtype=bool)
            tmp_mask[sampl_ind] = True
            design_mask[~tmp_mask] = False

            x_s, y_s, col_s = x[sampl_ind], y1[sampl_ind], col[sampl_ind]
            pass_mask = col_s <= self.lim_factor
            ax.scatter(x_s[pass_mask], y_s[pass_mask], s=20, c='#4C72B0', marker=mrk_lst[i], alpha=0.6)
            ax.scatter(x_s[~pass_mask], y_s[~pass_mask], s=20, c='#DD8452', marker=mrk_lst[i], alpha=0.6)

            # _ = ax.scatter(x[sampl_ind], y1[sampl_ind], c=col[sampl_ind], marker=mrk_lst[i], cmap='coolwarm',
            #                vmin=vmin, vmax=vmax, label=regime.NAME, alpha=0.7)
            ax.scatter(x[design_mask], y1[design_mask], s=40, facecolors='none', edgecolors='black', marker='o', alpha=0.4)
            ax.scatter([], [], c='black', marker=mrk_lst[i], alpha=0.5, label=regime.NAME)
            begin = len(x)

        ax.scatter([], [], facecolors='none', edgecolors='black', marker='o', alpha=0.5, label='Fires near risk target')
        ax.scatter([], [], c='#4C72B0', marker='s', alpha=0.5, label='Less severe than risk target')
        ax.scatter([], [], c='#DD8452', marker='s', alpha=0.5, label='More severe than risk target')

        ax.set_ylim([0, ax.get_ylim()[1] + 200])
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_xlabel('Fire duration (min)')
        ax.set_ylabel('Maximum gas temperature (°C)')
        ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 100))
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 100))
        ax.legend()
        # cb = fig.colorbar(_, extend='both')
        # cb.set_label('Maximum element temperature (°C)')
        # cb.ax.plot([0, 3000], [self.lim_factor]*2, 'black', alpha=0.4)
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
            list_of_inputs=['q_f_d', 'Q', 't_lim', 'spr_rate', 'flap_angle', 'T_nf_max', 'T_amb'],
            filename='fire_params')
        self._plot_convergence_study()
        self._plot_max_el_temp_duration_quick()

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
            list_of_inputs=['q_f_d', 'Q', 't_lim', 'spr_rate', 'flap_angle', 'T_nf_max', 'T_amb'],
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
        if self.analysis_type is 'quick':
            report = {'name': self.name,
                      'ID': self.ID,
                      'eqv_est': self.outputs['eqv_req'],
                      'eqv_low': self.outputs['eqv_req_conf'][0],
                      'eqv_high': self.outputs['eqv_req_conf'][1],
                      'success_conv': self.outputs['success_conv'],
                      'n_itr': self.optm_result.nfev,
                      'itr_err': self.optm_result.fun}
        elif self.analysis_type is 'full':
            report = {'name': self.name,
                      'ID': self.ID,
                      'eqv_est': self.outputs['eqv_req'],
                      'eqv_low': self.outputs['eqv_req_conf'][0],
                      'eqv_high': self.outputs['eqv_req_conf'][1],
                      'success_conv': True}

        return report

class CaseControler:

    def __init__(self, inputs, out_f):
        self.risk_method = None
        self.eqv_method = None
        self.mc_method = None
        self.cases = {}
        self.case_reports = []
        self.run_a_summary = None
        self.run_b_summary = None

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

    def conduct_pre_run_calculations(self):
        # calculate risk target
        self.risk_method.assess_risk_target()
        # calculate eqv_protection
        self.eqv_method.get_equivalent_protection()
        self.eqv_method.report_eqv_data(save_loc=self.out_f)
        #TODO plot equivalent protection

    def _get_cases(self):

        self.inputs['parameters'] = collections.OrderedDict(sorted(self.inputs['parameters'].items()))
        tmp_list_inputs = [self.inputs['parameters'][k] for k in self.inputs['parameters']]
        tmp_list_numeric_mask = [list(range(1, len(k) + 1)) for k in tmp_list_inputs]

        if self.inputs['param_mode'] == 'grid':
            all_cases = itertools.product(*tmp_list_inputs)
            all_cases_numeric_mask = itertools.product(*tmp_list_numeric_mask)
            for i, (case, num_mask) in enumerate(zip(all_cases, all_cases_numeric_mask)):
                self.cases[f'{(i+1):03d}'] = {
                    'label': CaseControler._create_case_name(
                        params=self.inputs['parameters'],  num_mask=num_mask, input_num_map=tmp_list_numeric_mask),
                    'params': {label: case for (label, case) in zip(self.inputs['parameters'], case)}}
        elif self.inputs['param_mode'] == 'one_at_a_time':
            all_cases = CaseControler._generate_one_at_a_time_combs(tmp_list_inputs)
            all_cases_numeric_mask = CaseControler._generate_one_at_a_time_combs(tmp_list_numeric_mask)
            for i, (case, num_mask) in enumerate(zip(all_cases, all_cases_numeric_mask)):
                self.cases[f'{(i+1):03d}'] = {
                    'label': CaseControler._create_case_name(
                        params=self.inputs['parameters'],  num_mask=num_mask, input_num_map=tmp_list_numeric_mask),
                    'params': {label: case for (label, case) in zip(self.inputs['parameters'], case)}}
        elif self.inputs['param_mode'] == 'parallel':
            for i in range(min([len(k) for k in tmp_list_inputs])):
                case = [value[i] for value in tmp_list_inputs]
                self.cases[f'{(i + 1):03d}'] = {
                    'label': f'input_{i+1}',
                    'params': {label: case for (label, case) in zip(self.inputs['parameters'], case)}}

    @staticmethod
    def _create_case_name(params, num_mask, input_num_map):
        """Static method for creating case label.
        Inputs:
            params (list or dict): list containing the input labels
            num_mask (list): contains the combination of inputs for the specific case mapped to the order of rows
            defined in the UI. Elements in the list must follow the order of parameters in params
            input_num_map (list of lists): list of list containing the inputs row number (starting from 1) as defined in the UI
            it must be in the same order as the labels in params
        Reruns:
            Label name (as str) in the form of {ui_label}{order}_* for only the parameters that are changed between
            different cases"""

        tmp = [f'{cfg.UNIT_CATALOGUE[key]["ui_label"]}{j}' for (key, j, m) in zip(params, num_mask, input_num_map) if len(m) != 1]
        return '_'.join(tmp)

    @staticmethod
    def _generate_one_at_a_time_combs(list_of_iterables):
        """Generates combinations from a list of iterables. The first combination, i.e the basecase, is formed from
        the first element of each iterable. The following combinations are generated by varying only one element of
        each iterable at a time while preserving the other elements of the combination as per the base case

        Inputs:
            list_of_iterables (array like of iterables): list of iterables to be combined. Lengths must be greater
            than zero
        Returns:
            combs (array like of iterables): All combinations as per the description. First element is the base case."""

        combs = [[i[0] for i in list_of_iterables]]  # defines the basecase
        for i, data in enumerate(list_of_iterables):
            for param in data[1:]:
                combs.append([param if i == j else k[0] for j, k in enumerate(list_of_iterables)])
        return combs

    def run_a_study(self):
        self._get_cases()
        for i, case_ID in enumerate(self.cases):
            print(f'{i+1}/{len(self.cases)}. Analysing case {self.cases[case_ID]["label"]}.')
            self.case = AssessmentCase(
                name=self.cases[case_ID]['label'],
                ID=case_ID,
                input_defs=self.cases[case_ID]['params'],
                risk_model=self.risk_method,
                mc_engine=self.mc_method,
                ht_model=self.eqv_method,
                heating_regimes_inputs=self.inputs['heating_regimes'],
                save_loc=os.path.join(self.out_f, 'run_a'),
                analysis_type='quick',
                sample_size=self.inputs['run_a_sample_size'],
                bootstrap_rep=self.inputs['bootstrap_rep'])
            self.case.run_analysis()
            print(f'Case {self.case.ID}_{self.case.name} initialised successfully.')
            print(f'Analysis for {self.case.name} completed. Convergence status: {self.case.outputs["success_conv"]}')
            print(f'Assessed equivalence: {self.case.outputs["eqv_req"]:.0f}, conf: {self.case.outputs["eqv_req_conf"].round(1)}\n')
            self.case_reports.append(self.case.report_to_main())
            if i == 1000:
                break

        self._summarise_run(which='run_a')
        self._plot_summary_bars(which='run_a')

    def _summarise_run(self, which):
        """Creates summary datatable with case reports
        Inputs:
            which (str): Which run to summarise. Value should be either 'run_a' or 'run_b."""

        data = pd.DataFrame(self.case_reports)
        if which == 'run_a':
            self.run_a_summary = data
            data.copy().astype(str).to_csv(os.path.join(self.out_f, 'run_a', 'cases_summary.csv'), index=False)
        elif which == 'run_b':
            self.run_b_summary = data
            data.copy().astype(str).to_csv(os.path.join(self.out_f, 'run_b', 'cases_summary.csv'), index=False)

    def _plot_summary_bars(self, which):

        if which == 'run_a':
            save_loc = 'run_a'
            data = self.run_a_summary
        elif which == 'run_b':
            save_loc = 'run_a'
            data = self.run_b_summary

        data = data.sort_values(by='eqv_est').reset_index()
        err_low = np.absolute((data['eqv_est'] - data['eqv_low']).values)
        err_high = np.absolute((data['eqv_est'] - data['eqv_high']).values)

        sns.set()
        fig, ax = plt.subplots(figsize=(10, 0.33 * len(data) + 3))
        y_pos = data.index.values

        idx_conv = data[data['success_conv']].index
        idx_not_conv = data[~data['success_conv']].index

        ax.barh(y_pos[idx_conv], data.loc[idx_conv, 'eqv_est'],
                xerr=[err_low[idx_conv], err_high[idx_conv]], ecolor='black', capsize=4, alpha=0.3, )
        ax.plot(data.loc[idx_conv, 'eqv_est'], y_pos[idx_conv], 'o', color='#4C72B0', alpha=0.9, label='Assessed case')
        if len(idx_not_conv) != 0:
            ax.barh(y_pos[idx_not_conv], data.loc[idx_not_conv, 'eqv_est'],
                    xerr=[err_low[idx_not_conv], err_high[idx_not_conv]], ecolor='#DD8452', alpha=0.3, )
            ax.plot(data.loc[idx_not_conv, 'eqv_est'], y_pos[idx_not_conv], 'o', color='#DD8452', alpha=0.9,
                    label='Case did not converge')

        ax.plot([], [], '-', color='black', alpha=0.9, label='95% confidence')
        ax.set_yticks(y_pos)
        start, end = ax.get_xlim()
        start = ((data['eqv_est'].min() - 10) // 10) * 10
        ax.set_xlim([start, end])
        ax.set_xticks(np.arange(start, end, 5))
        ax.set_yticklabels(data['ID'].astype(str) + '_' + data['name'])
        ax.set_xlabel('Equivalent severity (min)')
        plt.legend()

        plt.savefig(os.path.join(self.out_f, save_loc, f'results_summary.png'),
                    dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    def process_a_study_results(self):
        pass

    def run_b_study(self):
        pass

