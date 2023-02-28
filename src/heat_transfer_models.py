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

    def __init__(self, equivalent_curve, sect_prop, mat_prop, prot_prop, T_lim, eqv_max):
        super().__init__(equivalent_curve, sect_prop, mat_prop, prot_prop, T_lim, eqv_max)
        self.label = 'Steel EC3 HT'
        self.descr = '1D heat transfer in accordance with BS EN 1993-1-2'


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

        A = 2*sect_prop['fl_t']*sect_prop['B'] + (sect_prop['D'] - 2*sect_prop['fl_t'])*sect_prop['wb_t']
        if sect_prop['exp_sides'] == 'four':
            P = 2*sect_prop['D'] + 4*sect_prop['B'] - 2*sect_prop['wb_t']
        elif sect_prop['exp_sides'] == 'three':
            P = 2 * sect_prop['D'] + 3 * sect_prop['B'] - 2 * sect_prop['wb_t']
        return {'Av': P/A}


    def get_equivelant_protection(self):
        pass
