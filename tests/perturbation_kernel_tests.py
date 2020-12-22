import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.exchange_mcmc import TruncnormPerturbationKernel


class testTruncnormPerturbationKernel(unittest.TestCase):

    def setUp(self):
        self.seed = 10
        self.kernel = TruncnormPerturbationKernel(lower_bound=np.ones(4), upper_bound=np.ones(4) * 3)
        self.kernel_1d = TruncnormPerturbationKernel(lower_bound=np.ones(1), upper_bound=np.ones(1) * 3)
        self.kernel_2d = TruncnormPerturbationKernel(lower_bound=np.ones(2), upper_bound=np.ones(2) * 3)
        self.rng = np.random.RandomState(self.seed)

    def test_generate(self):
        x = self.kernel.sample(old_position=np.zeros(4), sigma=np.arange(4) + 1, rng=self.rng)
        # print(x)

    def test_log_pdf(self):
        # check if computation works:
        x = self.kernel.sample(old_position=np.zeros(4), sigma=1, rng=self.rng)
        log_pdf = self.kernel.log_pdf(old_position=np.zeros(4), new_position=x, sigma=1)
        log_pdf_2 = self.kernel.log_pdf(old_position=np.zeros(4), new_position=np.ones(4), sigma=1)
        log_pdf_3 = self.kernel.log_pdf(old_position=np.zeros(4), new_position=np.ones(4) * 3, sigma=1)
        self.assertTrue(log_pdf < log_pdf_2)  # necessary condition
        self.assertTrue(log_pdf > log_pdf_3)  # necessary condition

    def test_pdf_agrees_hist(self):
        samples = np.zeros(1000)
        for i in range(1000):
            samples[i] = self.kernel_1d.sample(old_position=np.ones(1) * 1.5, sigma=1, rng=self.rng)
        x_linspace = np.linspace(1, 3)
        log_pdf = np.zeros_like(x_linspace)
        for i in range(len(x_linspace)):
            log_pdf[i] = self.kernel_1d.log_pdf(1.5, x_linspace[i], sigma=1)
        # a, b = (1 - 0) / 1, (3 - 0) / 1
        # log_pdf = truncnorm.logpdf(x_linspace, a, b, loc=0, scale=1)
        plt.plot(x_linspace, np.exp(log_pdf))
        plt.hist(samples, density=True, alpha=0.5)
        # plt.show()  # show this plot when running this test independently only

    def test_pdf_agrees_hist_2d(self):
        samples = np.zeros((100000, 2))
        for i in range(100000):
            samples[i] = self.kernel_2d.sample(old_position=np.ones(2) * 1.5, sigma=1, rng=self.rng)
        x_linspace = np.linspace(1, 3)
        y_linspace = np.linspace(1, 3)
        X, Y = np.meshgrid(x_linspace, y_linspace)
        log_pdf = np.zeros((x_linspace.shape[0], y_linspace.shape[0]))
        for i in range(len(x_linspace)):
            for j in range(len(y_linspace)):
                log_pdf[i, j] = self.kernel_2d.log_pdf(np.ones(2) * 1.5, np.array([x_linspace[i], y_linspace[j]]),
                                                       sigma=1)
        # print(log_pdf.shape)
        fig, ax = plt.subplots(1, 2)
        ax[0].contourf(X, Y, np.exp(log_pdf), 20, cmap='Blues')
        ax[1].hist2d(samples[:, 0], samples[:, 1], density=True, alpha=0.5, bins=15, cmap='Blues')
        # plt.show()  # show this plot when running this test independently only
