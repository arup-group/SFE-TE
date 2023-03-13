"""Holds distribution information"""

import numpy as np
from scipy.interpolate import interp1d

class GenericDistr:

    def __init__(self, params, generator):
        self.rnd = generator
        self._unpack_params(params)

    def _unpack_params(self):
        raise NotImplemented

    def calc_params(self):
        raise NotImplemented

    def sample(self, sample_size):
        raise NotImplemented

    def draw(self):
        raise NotImplemented


class TriangularDistr(GenericDistr):
    def __init__(self, params, generator):
        super().__init__(params, generator)
        self.label = 'Triangular distribution'

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r,alpha, A], where l is the left bound, r is the right bound,
            gamma (0 to 1) is interval fraction and A (0, 1) is fraction of samples within the interval fraction"""

        self.l, self.r, self.alpha, self.A = params[0], params[1], params[2], params[3]
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError(f'Incorrect input for {self.label}. P1 = {self.alpha} must be between 0 and 1.')
        if self.A > 1 or self.A < 0:
            raise ValueError(f'Incorrect input for {self.label}. P2 = {self.A} must be between 0 and 1.')

    def _calc_mode(self):
        """Calculates mode of triangular distribution. See TGN for derivation"""
        self.m = self.alpha**2*(self.r - self.l)/self.A + self.l
        if self.m > self.r or self.m < self.l + self.alpha*(self.r-self.l) or self.m < self.l:
            self.m = self.r - (self.r - self.l)*(1 - self.alpha)**2 / (1 - self.A)
            if self.m < self.l or self.m > self.r or self.m > self.l + self.alpha*(self.r-self.l):
                raise ValueError(
                    f'{self.label} with mode of {self.m} invalid for min = {self.l}, max = {self.r}, P1 = {self.alpha}, and P2 = {self.A}')

    def calc_params(self):
        self._calc_mode()

    def sample(self, sample_size):
        return self.rnd.triangular(left=self.l, mode=self.m, right=self.r, size=sample_size)

    def draw(self):
        l_lim = self.l - 0.1*(self.r-self.l)
        r_lim = self.r + 0.1 * (self.r - self.l)
        peak = 2/(self.r-self.l)
        return np.array([[l_lim, 0], [self.l, 0], [self.m, peak], [self.r, 0], [r_lim, 0]])


class UniBiomodalDistr(GenericDistr):
    def __init__(self, params, generator):
        super().__init__(params, generator)
        self.label = 'Bimodal uniform distribution'

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r,alpha, A], where l is the left bound, r is the right bound,
            gamma (0 to 1) is interval fraction and A (0, 1) is fraction of samples within the interval fraction"""

        self.l, self.r, self.alpha, self.A = params[0], params[1], params[2], params[3]
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError(f'Incorrect input for {self.label}. P1 = {self.alpha} must be between 0 and 1.')
        if self.A > 1 or self.A < 0:
            raise ValueError(f'Incorrect input for {self.label}. P2 = {self.A} must be between 0 and 1.')

    def _calc_mode(self):
        """Calculates modes of uniform bimodal distribution. See TGN for derivation"""
        self.h1 = self.A/(self.r - self.l)/self.alpha
        self.h2 = (1 - self.A) / (self.r - self.l) / (1 - self.alpha)
        self.m = self.l + self.alpha*(self.r - self.l)

    def calc_params(self):
        self._calc_mode()

    def sample(self, sample_size):

        #Define quantile interpolation function
        pts = np.array([[0, self.l], [self.A, self.m], [1, self.r]])
        f = interp1d(pts[:, 0], pts[:, 1])

        return f(self.rnd.uniform(0, 1, size=sample_size))

    def draw(self):

        l_lim = self.l - 0.1 * (self.r - self.l)
        r_lim = self.r + 0.1 * (self.r - self.l)

        return np.array(
            [[l_lim, 0], [self.l, 0], [self.l, self.h1], [self.m, self.h1], [self.m, self.h2],  [self.r, self.h2], [self.r, 0],  [r_lim, 0]])


class ProbControl:

    SUPPORTED_DISTR = {
        'uniform': TriangularDistr,
        'uni_bimodal': UniBiomodalDistr,
        'triangular': TriangularDistr,
        'weibull': TriangularDistr,
        'user_def': TriangularDistr,
        'gumbel': TriangularDistr,
        'fixed_point': TriangularDistr}

    def __init__(self, inputs, seed):
        self.inputs = inputs
        self.rnd = ProbControl._initiate_random_seed(seed)
        self.sampled_inputs = {'values': {}, 'curves': {}}
        self.input_distr_curves = {}

    @staticmethod
    def _initiate_random_seed(seed):
        return np.random.default_rng(seed)

    def sample_inputs(self, sample_size):
        """Samples requested inputs"""

        for input in self.inputs:
            distr = ProbControl.SUPPORTED_DISTR[self.inputs[input][0]](params=self.inputs[input][1:], generator=self.rnd)
            distr.calc_params()
            self.sampled_inputs['values'][input] = distr.sample(sample_size=sample_size)
            self.sampled_inputs['curves'][input] = distr.draw()
        return self.sampled_inputs

    def draw_distr_curves(self):
        pass