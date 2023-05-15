import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class GenericRiskModel:

    def __init__(self):
        self.risk_target = None

    def _check_inputs(self):
        raise NotImplementedError

    def assess_risk_target(self):
        raise NotImplementedError

    def risk_sensitivity_study(self):
        raise NotImplementedError


class KirbyBS9999(GenericRiskModel):

    RISK_FACTORS = {'awake': 64.8,
                    'sleeping': 25.92}

    def __init__(self, sprinkler_reliability, building_height, occupancy):
        super().__init__()
        self.sprinkler_reliability = sprinkler_reliability
        self.building_height = building_height
        self.occupancy = occupancy

        #Parameters that will change
        self.total_reliability = None
        self.struct_reliability = None

        self._check_inputs()

    def _check_inputs(self):

        if self.sprinkler_reliability < 0 or self.sprinkler_reliability >= 100:
            raise ValueError(f'Incorrect sprinkler reliability value of {self.sprinkler_reliability}.\n'
                             f'Sprinkler reliability should be in the interval [0, 100). ')
        if self.building_height <= 0:
            raise ValueError(f'Incorrect building height value of {self.building_height}.\n'
                             f'Building height must be a positive value. ')

    def assess_risk_target(self):
        """Assess total and structural reliability as per methodology outlined in Kirby et al"""

        self.total_reliability = 1 - KirbyBS9999.RISK_FACTORS[self.occupancy]/self.building_height**2
        self.struct_reliability = (self.total_reliability - self.sprinkler_reliability/100)/(1 - self.sprinkler_reliability/100)
        self.risk_target = self.struct_reliability


    def _sprinkler_sensitivity(self, analysis_case):
        """Performs sprinkler sensitivity study"""

        spr_rel_range = np.arange(0, 100, 1)/100
        struct_rel = (self.total_reliability - spr_rel_range)/(1 - spr_rel_range)
        spr_rel_range = spr_rel_range[struct_rel > 0]
        struct_rel = struct_rel[struct_rel > 0]

        reliability_curve = analysis_case.outputs['reliability_curve']
        conf_curve = analysis_case.outputs['reliability_conf']

        #compute reliability
        f = interpolate.interp1d(reliability_curve[:, 2], reliability_curve[:, 0], fill_value=-1, bounds_error=False)
        reliability = f(struct_rel)

        #calc confidence at target by interpolation
        conf = [0, 0]
        for i in range(2):
            f = interpolate.interp1d(conf_curve[:, i], reliability_curve[:, 0], fill_value=-1, bounds_error=False)
            conf[i] = f(struct_rel)

        sns.set()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(spr_rel_range[reliability != -1]*100, reliability[reliability != -1],
                color='black',
                label='Target severity requirement')
        ax.plot(spr_rel_range[conf[0] != -1]*100, conf[0][conf[0] != -1],
                color='grey',
                linestyle='dashed',
                label='95% conf. interval',
                linewidth=1)
        ax.plot(spr_rel_range[conf[1] != -1]*100, conf[1][conf[1] != -1],
                color='grey',
                linestyle='dashed',
                linewidth=1)

        ax.set_xlabel('Sprinkler reliability (%)')
        ax.set_ylabel('Exposure rating (min)')
        ax.legend()

        # Save figure
        plt.savefig(os.path.join(analysis_case.save_loc, f'{analysis_case.ID}_sprinkler_sensitivity.png'),
                    bbox_inches="tight",
                    dpi=150)
        # Save data
        data = pd.DataFrame({'spr_reliability': spr_rel_range,
                             'fire_severity_req': reliability,
                             'low_bound': conf[1],
                             'upper_bound': conf[0]})
        data.round(3).to_csv(os.path.join(analysis_case.save_loc, 'data', f'{analysis_case.ID}_sprinkler_sensitivity.csv'), index=False)
        plt.close(fig)


    def risk_sensitivity_study(self, **kwargs):
        self._sprinkler_sensitivity(**kwargs)
