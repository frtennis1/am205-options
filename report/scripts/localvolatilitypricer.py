import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import scipy.interpolate as interp
from sklearn.preprocessing import PolynomialFeatures
from copy import copy

# Requirements in notebook
# - Requires sampleendprice, price_calls, and InterpolatedLocalVol
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# %run report / scripts / sampleendprice.py
# %run report / scripts / pricecall.py
# %run interpolated_local_vol.py

class LocalVolatilityPricer:
    """An object that saves true prices and handles fitting and prediction"""

    def __init__(self, local_vol, expiries, strikes):
        """
        Instantiate a LocalVolatilityPricer object

        Inputs
        ------
        local_vol: float, float -> float, vectorized
            local volatility function at a given spot price and time

        expiries: np.array
            the dates of expiries simulated

        strikes: np.array
            the strike prices interested
        """
        self.local_vol = local_vol
        self.expiries = expiries
        self.strikes = strikes

    def get_true_prices(self, S0=100, interval_per_year=100, n_samples=int(1e5)):
        """
        Perform Monte Carlo pricing to simulate true prices of a call option

        Inputs
        ------
        S0: float
            initial price of the underlying

        interval_per_year: int
            the number of intervals per year simulated; dt = 1/intervals_per_year

        n_samples: int
            the number of samples simluted
        """
        self.S0 = S0
        true_prices = 0

        # price 100 times and take the average price to reduce sampling error
        for _ in tqdm(range(100)):
            sample = sample_end_prices(self.S0, self.local_vol, self.expiries,
                                       interval_per_year, n_samples)
            true_prices += price_calls(self.strikes, sample)

        self.true_prices = true_prices / 100
        return self.true_prices

    def fit_local_vol(self, fitter, interp_option='linear', *args, **kwargs):
        """
        Perform fitting using a given fitter

        Inputs
        ------
        fitter: function
            fitter's first argument takes in a DataFrame of true prices
            fitter returns a tuple (local_volatility, dCdT, d2CdK2) estimated

        interp_option: string
            option for interpolation passed to _interpolate_local_vol
            'linear' means linear interpolation
            'nn' means using scipy.interpolate.NearestNDInterpolator
        """

        lvol, dCdT, d2CdK2 = fitter(self.true_prices, *args, **kwargs)
        self.fitted_local_vol = lvol
        self.dCdT = dCdT
        self.d2CdK2 = d2CdK2
        self._interpolate_local_vol(interp_option)
        return self.fitted_local_vol

    def _interpolate_local_vol(self, interp_option):
        """Interpolate the grid of local volatility values using interp_option"""
        if interp_option == 'linear':
            data = self.fitted_local_vol.T.stack().reset_index().as_matrix()
            stks, exps, lv = data[:, 0], data[:, 1], data[:, 2]
            self.interp_fitted_local_vol = InterpolatedLocalVol(lv, stks, exps)
        elif interp_option == 'nn':
            data = self.fitted_local_vol.T.stack().reset_index().as_matrix()
            args, lv = data[:, :2], data[:, 2]
            self.interp_fitted_local_vol = interp.NearestNDInterpolator(
                args, lv)
        else:
            raise ValueError('interp_option not one of "linear", "nn"')

    def generate_prediction(self, interval_per_year=100, n_samples=int(1e5)):
        """
        Use interpolated local volatility estimates to compute option prices
        via Monte Carlo pricing
        """
        f_samples = sample_end_prices(self.S0, self.interp_fitted_local_vol,
                                      self.expiries, interval_per_year, n_samples)
        self.predicted_prices = price_calls(self.strikes, f_samples)
        return self.predicted_prices

    def plot_errors(self):
        """Plot errors obtained by comparing predicted prices to true prices"""
        f, ax = plt.subplots(nrows=2, figsize=(5, 2 * 5 / 1.618))
        sns.heatmap(self.predicted_prices
                    - self.true_prices, ax=ax[0], fmt='%.2f')
        sns.heatmap(np.abs(self.predicted_prices - self.true_prices), ax=ax[1])

    def plot_fit(self, time, xs=np.arange(70, 130, .1)):
        """Plot comparison of local vol estimates against true local vol"""
        plt.plot(xs, self.local_vol(xs, time))
        plt.plot(xs, self.interp_fitted_local_vol(xs, time))
