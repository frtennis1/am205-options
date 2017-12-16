import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

def sample_end_price(S0, local_vol_f, duration, n_intervals, n_samples):
    """
    Draw samples from the end price of an asset diffusion.

    Inputs
    ------
    S0 : float
        The initial spot price of the underlying at time t=0

    local_vol_f : float -> float (vectorized)
        The local volatility at a given spot price (assumed constant over time)

    duration : float
        The time to expiry, i.e. T.

    n_intervals : float
        Number of intervals into which to break up the numerical simulation

    n_samples : int
        Number of simulations to run

    Output
    -----
    S : NumPy float vector of length n_samples
        The ending spot prices of the asset diffusion for each simulation
    """
    if duration == 0:
        return S0 * np.ones(n_samples)


    dt = duration / n_intervals
    scaling_factor =  np.sqrt(duration / n_intervals)
    S = S0 * np.ones(n_samples)
    # Z = np.sqrt(dt) * np.random.randn(n_samples, n_intervals)
    for i in (range(1,n_intervals+1)):
        time = i / duration
        local_vols = local_vol_f(S, time).flatten()
        # S = S +  S * local_vols * Z[:, i - 1]
        growth_factor = np.exp(local_vols * np.random.randn(n_samples) * scaling_factor
                               - dt * local_vols**2/2)
        S = S * growth_factor
    return S


def sample_end_prices(S0, local_vol_f, durations, intervals_per_year, n_samples):
    S = np.zeros((n_samples, len(durations)))
    Dt = np.hstack([durations[0], np.diff(durations)])
    S_initial = S0
    for i, T in tqdm(enumerate(Dt)):
        start = durations[i]
        def local_vol_f_shifted(x,t):
            return local_vol_f(x, t+start)
        n_intervals = max(1, int(T * intervals_per_year))
        S_initial = sample_end_price(S_initial, local_vol_f, T, n_intervals, n_samples)
        S[:, i] = S_initial

    df = pd.DataFrame(S, columns=durations)
    df.columns.name = 'Expiries'
    return df
