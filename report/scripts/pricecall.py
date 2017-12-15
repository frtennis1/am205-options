import numpy as np
import pandas as pd

def price_call(K, end_samples):
    """
    Price a call option using Monte Carlo samples.
    
    Inputs
    ------
    K : float:
        The strike price of the call option we are pricing.
    
    end_samples : np.array[float]
        A NumPy array of draws from the distribution of prices
        of the underlying at expiry.
        
    Outputs
    -------
    px : float
        A point estimate for the price of the call option
    sd : float
        An estimate of the standard error incurred from this pricing.
    """
    
    payoffs = np.clip(np.subtract.outer(end_samples, K), 0, np.inf)
    px = payoffs.mean(axis=0)
    sd = payoffs.std(axis=0) / np.sqrt(len(end_samples))
    return px, sd

def price_calls(Ks, samples):
    payoffs = np.clip(np.subtract.outer(samples, Ks), 0, np.inf)
    px = payoffs.mean(axis=0)
    df = pd.DataFrame(px, index=samples.columns, columns=Ks)
    df.index.name = 'Expiry'
    df.columns.name = 'Strikes'
    return df