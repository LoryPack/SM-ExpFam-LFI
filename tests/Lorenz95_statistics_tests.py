import unittest
from time import time

import numpy as np

from src.utils_Lorenz95_example import LorenzLargerStatistics
from src.utils_Lorenz95_example import StochLorenz95


class test_statistics(unittest.TestCase):

    def setUp(self):
        self.theta1 = 2
        self.theta2 = 0.15
        self.sigma_e = 1
        self.phi = 0.4

        self.lorenz = StochLorenz95([self.theta1, self.theta2, self.sigma_e, self.phi], time_units=4,
                                    n_timestep_per_time_unit=30, name='lorenz')
        self.stat_large = LorenzLargerStatistics(degree=1, cross=False)

        self.rng = np.random.RandomState(seed=2)

        # generate from model:
        start = time()
        self.data = self.lorenz.forward_simulate([self.theta1, self.theta2, self.sigma_e, self.phi], 1, rng=self.rng)
        print(f"Generation took {time() - start:.4f} seconds")

    def test_stat(self):
        # compute statistics:
        start = time()
        statistics_new = self.stat_large.statistics(self.data)
        print(f"Statistics computation took {time() - start:.4f} seconds")

        # print(statistics_new)
        # check if some elements in the statistics_new vector are repeated:
        vec, counts = np.unique(statistics_new, return_counts=True)
        assert np.all(counts == 1)

    def test_invariant_to_data_permutation(self):
        permutation = self.rng.choice(40, 40, replace=False)
        cyclic_shift = np.roll(np.arange(40), shift=5)
        data_permutated = [self.data[0].reshape(40, -1)[permutation].reshape(1, -1)]
        data_cyclic_shifted = [self.data[0].reshape(40, -1)[cyclic_shift].reshape(1, -1)]
        # the moments and autocovariances should be invariant to any data permutation;
        # the other components should instead be invariant to cyclic shift

        statistics_new = self.stat_large.statistics(self.data)[0]
        statistics_new_perm = self.stat_large.statistics(data_permutated)[0]
        statistics_new_cyclic = self.stat_large.statistics(data_cyclic_shifted)[0]
        # The first 8 elements are permutation invariant
        assert np.allclose(statistics_new[:8], statistics_new_perm[:8])
        # All elements are invariant to cyclic shift
        assert np.allclose(statistics_new, statistics_new_cyclic)
