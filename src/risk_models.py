

class GenericRiskModel:

    def __init__(self):
        self.risk_target = None

    def _check_inputs(self):
        raise NotImplementedError

    def assess_risk_target(self):
        raise NotImplementedError

    def risk_sensitivity_study(self):
        raise NotImplementedError


class Kirby(GenericRiskModel):

    def __init__(self, sprinkler_reliability, building_height):
        super().__init__()
        self.sprinkler_reliability = sprinkler_reliability
        self.building_height = building_height

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
                             f'Building height must be a posstive value. ')

    def assess_risk_target(self):
        """Assess total and structural reliability as per methodology outlined in Kirby et al"""

        self.total_reliability = 1 - 64.8/self.building_height**2
        self.struct_reliability = (self.total_reliability - self.sprinkler_reliability/100)/(1 - self.sprinkler_reliability/100)


    def _sprinkler_sensitivity(self, reliability_curve, plot_debug, save_loc):
        """Performs sprinkler sensitivity study"""
        pass

    def risk_sensitivity_study(self, **kwargs):
        self._sprinkler_sensitivity(**kwargs)
