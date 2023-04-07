"""
This module contains standard heating regimes used for time equivalence
"""

import numpy as np

class GenericCurve:
    """Generic class"""

    def __init__(self):
        self.label = 'Generic'
        self.descr = 'Generic descr'

    def get_temp(self):
        raise NotImplemented

    def get_delta_temp(self):
        raise NotImplemented


class Iso834(GenericCurve):
    """Standard fire curve"""

    def __init__(self):
        self.label = 'ISO 834'
        self.descr = 'Standard fire heating curve to ISO 834'

    def get_temp(self, t):
        """ Gets a temperature reading(s) from ISO 834 curve. UNIT TEST REQUIRED.
        Inputs:
            t(float, array-like): time in [s]
        Returns:
            T(float, array-like): Temperature in [degC] """

        if isinstance(t, list):
            t = np.asarray(t)
        return 20 + 345*np.log10(8*t/60+1)

    def get_delta_temp(self, t1, t2):

        """Gets difference in temperature at two points in time
         where t2>t1 """

        return self.get_temp(t2) - self.get_temp(t1)




