"""Holds distribution information"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from scipy.special import gamma

class GenericDistr:

    def __init__(self, params, generator):
        self.rnd = generator
        self._unpack_params(params)

    def _unpack_params(self):
        raise NotImplementedError

    def calc_params(self):
        raise NotImplementedError

    def sample(self, sample_size):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError


class TriangularDistr(GenericDistr):
    label = 'Triangular distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r,alpha, A], where l is the left bound, r is the right bound,
            gamma (0 to 1) is interval fraction and A (0, 1) is fraction of samples within the interval fraction"""

        self.l, self.r, self.alpha, self.A = params[0], params[1], params[2], params[3]
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError(f'Incorrect input for {TriangularDistr.label}. P1 = {self.alpha} must be between 0 and 1.')
        if self.A > 1 or self.A < 0:
            raise ValueError(f'Incorrect input for {TriangularDistr.label}. P2 = {self.A} must be between 0 and 1.')

    def _calc_mode(self):
        """Calculates mode of triangular distribution. See TGN for derivation"""
        self.m = self.alpha**2*(self.r - self.l)/self.A + self.l
        if self.m > self.r or self.m < self.l + self.alpha*(self.r-self.l) or self.m < self.l:
            self.m = self.r - (self.r - self.l)*(1 - self.alpha)**2 / (1 - self.A)
            if self.m < self.l or self.m > self.r or self.m > self.l + self.alpha*(self.r-self.l):
                raise ValueError(
                    f'{TriangularDistr.label} with mode of {self.m} invalid for min = {self.l}, max = {self.r}, P1 = {self.alpha}, and P2 = {self.A}')

    def calc_params(self):
        self._calc_mode()

    def sample(self, sample_size):
        return self.rnd.triangular(left=self.l, mode=self.m, right=self.r, size=sample_size)

    def draw(self):
        l_lim = self.l - 0.2*(self.r-self.l)
        r_lim = self.r + 0.2 * (self.r - self.l)
        peak = 2/(self.r-self.l)
        return np.array([[l_lim, 0], [self.l, 0], [self.m, peak], [self.r, 0], [r_lim, 0]])

    def report_params(self):
        return {'label': TriangularDistr.label, 'mode': self.m}



class UniBiomodalDistr(GenericDistr):
    label = 'Bimodal uniform distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r,alpha, A], where l is the left bound, r is the right bound,
            gamma (0 to 1) is interval fraction and A (0, 1) is fraction of samples within the interval fraction"""

        self.l, self.r, self.alpha, self.A = params[0], params[1], params[2], params[3]
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError(f'Incorrect input for {UniBiomodalDistr.label}. P1 = {self.alpha} must be between 0 and 1.')
        if self.A > 1 or self.A < 0:
            raise ValueError(f'Incorrect input for {UniBiomodalDistr.label}. P2 = {self.A} must be between 0 and 1.')

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
        l_lim = self.l - 0.2 * (self.r - self.l)
        r_lim = self.r + 0.2 * (self.r - self.l)
        return np.array(
            [[l_lim, 0], [self.l, 0], [self.l, self.h1], [self.m, self.h1], [self.m, self.h2],  [self.r, self.h2], [self.r, 0],  [r_lim, 0]])

    def report_params(self):
        return {'label': UniBiomodalDistr.label, 'mode': self.m, 'h1': self.h1, 'h2': self.h2}


class UniformDistr(GenericDistr):
    label = 'Uniform distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 2 element list, [l,r], where l is the left bound, r is the right bound."""
        self.l, self.r = params[0], params[1]

    def _calc_h(self):
        """Calculates height of the distribution"""
        self.h = 1/(self.r - self.l)

    def calc_params(self):
        self._calc_h()

    def sample(self, sample_size):
        """Samples from unifrom distribution"""
        return self.rnd.uniform(self.l, self.r, size=sample_size)

    def draw(self):
        l_lim = self.l - 0.2 * (self.r - self.l)
        r_lim = self.r + 0.2 * (self.r - self.l)
        return np.array(
            [[l_lim, 0], [self.l, 0], [self.l, self.h], [self.r, self.h],[self.r, 0],  [r_lim, 0]])

    def report_params(self):
        return {'label': UniformDistr.label}


class NormalDistr(GenericDistr):
    label = 'Normal distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r, mu. sigma], where l is the left bound, r is the right bound,
            mu is mean and sigma is standard deviation"""
        self.l, self.r, self.mu, self.sigma = params[0], params[1], params[2], params[3]

    def _calc_ends(self):
        """Calculates CDF(x) where x are truncated values if defined, i.e left inverse and right inverse.
        These are then used in reverse sampling routine"""

        if self.l is None:
            self.l_inv = 0
        else:
            self.l_inv = stats.norm.cdf(self.l, loc=self.mu, scale=self.sigma)

        if self.r is None:
            self.r_inv = 1
        else:
            self.r_inv = stats.norm.cdf(self.r, loc=self.mu, scale=self.sigma)

    def calc_params(self):
        self._calc_ends()

    def sample(self, sample_size):
        """Sample distribution using inverse sampling method"""
        return stats.norm.ppf(self.rnd.uniform(self.l_inv, self.r_inv, size=sample_size), loc=self.mu, scale=self.sigma)

    def draw(self):
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 100)
        y = stats.norm.pdf(x, loc=self.mu, scale=self.sigma)/(self.r_inv - self.l_inv)
        if self.l is not None:
            y[x<self.l] = 0
        if self.r is not None:
            y[x > self.r] = 0
        return np.array([x, y]).T

    def report_params(self):
        return {'label': NormalDistr.label, 'l_inv': self.l_inv, 'r_inv': self.r_inv,
                'loss': 1 - self.r_inv - self.l_inv}


class WeibullDistr(GenericDistr):
    label = 'Weibull distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)


    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r, mu, sigma], where l is the left bound, r is the right bound,
            mu is mean and sigma is standard deviation. Values must be positive"""
        if any([k < 0 for k in params if k is not None]):
            raise ValueError(f'Incorrect input for {WeibullDistr.label}. All input parameters must be positive.')
        self.l, self.r, self.mu, self.sigma = params[0], params[1], params[2], params[3]

    def _calc_k_lambd(self):
        """Calculates k and lambda factor from standard deviation and mean. Note this is approximate solution
        See https://journals.ametsoc.org/view/journals/apme/17/3/1520-0450_1978_017_0350_mfewsf_2_0_co_2.xml"""

        self.k = (self.sigma/self.mu)**(-1.086)
        self.lambd = self.mu/gamma(1+1/self.k)

    def _calc_ends(self):
        """Calculates CDF(x) where x are truncated values if defined, i.e left inverse and right inverse.
        These are then used in reverse sampling routine"""

        if self.l is None:
            self.l_inv = 0
        else:
            self.l_inv = stats.weibull_min.cdf(self.l, c=self.k, loc=0, scale=self.lambd)

        if self.r is None:
            self.r_inv = 1
        else:
            self.r_inv = stats.weibull_min.cdf(self.r, c=self.k, loc=0, scale=self.lambd)

    def calc_params(self):
        self._calc_k_lambd()
        self._calc_ends()

    def sample(self, sample_size):
        """Sample distribution using inverse sampling method"""
        return stats.weibull_min.ppf(self.rnd.uniform(self.l_inv, self.r_inv, size=sample_size), c=self.k, loc=0, scale=self.lambd)

    def draw(self):
        x = np.linspace(0, self.mu + 5*self.sigma, 100)
        y = stats.weibull_min.pdf(x, c=self.k, loc=0, scale=self.lambd)/(self.r_inv - self.l_inv)
        if self.l is not None:
            y[x < self.l] = 0
        if self.r is not None:
            y[x > self.r] = 0
        return np.array([x, y]).T

    def report_params(self):
        return {'label': WeibullDistr.label, 'l_inv': self.l_inv, 'r_inv': self.r_inv, 'loss': 1 - self.r_inv - self.l_inv,
                'k': self.k, 'lambda': self.lambd}


class GumbelDistr(GenericDistr):
    label = 'Gumbel distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): 4 element list, [l,r, mu, sigma], where l is the left bound, r is the right bound,
            mu is mean and sigma is standard deviation. Values must be positive"""

        self.l, self.r, self.mu, self.sigma = params[0], params[1], params[2], params[3]
        if self.sigma < 0:
            raise ValueError(f'Incorrect input for {GumbelDistr.label}. Variance must be positive.')

    def _calc_beta_m(self):
        """Calculates beta and z factor from standard deviation and mean. Note this is approximate solution.
        See https://en.wikipedia.org/wiki/Gumbel_distribution"""

        self.beta = self.sigma/np.pi*(6**0.5)
        self.m = self.mu - 0.5772*self.beta  # 0.57772 is Euler - Mascheroni constant

    def _calc_ends(self):
        """Calculates CDF(x) where x are truncated values if defined, i.e left inverse and right inverse.
        These are then used in reverse sampling routine"""

        if self.l is None:
            self.l_inv = 0
        else:
            self.l_inv = stats.gumbel_r.cdf(self.l, loc=self.m, scale=self.beta)

        if self.r is None:
            self.r_inv = 1
        else:
            self.r_inv = stats.gumbel_r.cdf(self.r, loc=self.m, scale=self.beta)

    def calc_params(self):
        self._calc_beta_m()
        self._calc_ends()

    def sample(self, sample_size):
        """Sample distribution using inverse sampling method"""
        return stats.gumbel_r.ppf(self.rnd.uniform(self.l_inv, self.r_inv, size=sample_size), loc=self.m, scale=self.beta)

    def draw(self):
        x = np.linspace(0, self.mu + 5*self.sigma, 100)
        y = stats.gumbel_r.pdf(x, loc=self.m, scale=self.beta)/(self.r_inv - self.l_inv)
        if self.l is not None:
            y[x < self.l] = 0
        if self.r is not None:
            y[x > self.r] = 0
        return np.array([x, y]).T

    def report_params(self):
        return {'label': GumbelDistr.label, 'l_inv': self.l_inv, 'r_inv': self.r_inv, 'loss': 1 - self.r_inv-self.l_inv,
                'beta': self.beta, 'm': self.m}


class FixedPointDistr(GenericDistr):
    label = 'Fixed point distribution'

    def __init__(self, params, generator):
        super().__init__(params, generator)

    def _unpack_params(self, params):
        """
        Inputs:
            params (array like): Requires only one parameter at 3rd poistion"""
        self.value = params[2]

    def calc_params(self):
        pass

    def sample(self, sample_size):
        """Sample distribution using inverse sampling method"""
        return np.full(shape=sample_size, fill_value=self.value, dtype=np.float64)

    def draw(self):
        return np.array([[self.value-0.02, 25], [self.value+0.02, 25]])

    def report_params(self):
        return {'label': FixedPointDistr.label}


class UserDefDistr(GenericDistr):
    label = 'User defined distribution'

    def __init__(self, params, generator):
        raise NotImplementedError(f'{UserDefDistr.label} feature not developed at current version.')


class ProbControl:

    SUPPORTED_DISTR = {
        'uniform': UniformDistr,
        'uni_bimodal': UniBiomodalDistr,
        'triangular': TriangularDistr,
        'weibull': WeibullDistr,
        'user_def': UserDefDistr, #TODO
        'gumbel': GumbelDistr,
        'fixed_point': FixedPointDistr,
        'normal': NormalDistr}

    def __init__(self, seed):

        self.rnd = ProbControl._initiate_random_seed(seed)
        self.sampled_inputs = {'values': {}, 'curves': {}, 'interim_params': {}}
        self.input_distr_curves = {}

    @staticmethod
    def _initiate_random_seed(seed):
        return np.random.default_rng(seed)

    def sample_inputs(self, inputs, sample_size):
        """Samples requested inputs.

            Inputs:
                sample_size (int): sample size
                inputs (dict): dictionary containing all inputs to be sampled. See TGN.

            Returns:
                sampled_inputs (dict): Dictionary with 3 keys: 'values' - contains 1 X sample size array of sampled
                values; 'curves' - contains data for plotting target analytical curve; 'interim_params' - contains data
                on calculated interim parameters for checking"""

        for input in inputs:
            distr = ProbControl.SUPPORTED_DISTR[inputs[input][0]](params=inputs[input][1:], generator=self.rnd)
            distr.calc_params()
            self.sampled_inputs['values'][input] = distr.sample(sample_size=sample_size)
            self.sampled_inputs['curves'][input] = distr.draw()
            self.sampled_inputs['interim_params'][input] = distr.report_params()

        return self.sampled_inputs

