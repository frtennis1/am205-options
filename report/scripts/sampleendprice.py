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
    dt = duration / n_intervals
    scaling_factor = duration / np.sqrt(n_intervals)
    S = np.zeros((n_samples, n_intervals+1))
    S[:,0] = S0
    for i in range(1,n_intervals+1):
        time = i / duration
        local_vols = local_vol_f(S[:,i-1], time)
        growth_factor = np.exp(local_vols * np.random.randn(n_samples) * scaling_factor
                              - dt * local_vols**2/2)
        S[:,i] = S[:,i-1] * growth_factor
    return S[:,-1]