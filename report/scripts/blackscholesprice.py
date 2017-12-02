def black_scholes_price(S, vol, T, K):
    """
    A Black Scholes pricer for European call options which assumes 
    interest rates are 0 and an underlying that doesn't pay out dividends.
    
    Inputs
    ------
    S : float
        The initial price of the underlying
    vol : float
        The annualized volatility of the underlying
    T : float
        The time to expiry (in years)
    K : float
        The strike price of the call option
        
    Outputs
    -------
    px : float
        The price of the call option under the Black Scholes model with 0
        risk-free interest rate.
    """
    d1 = (np.log(S / K) + (vol**2/2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return S * norm.cdf(d1) - K*norm.cdf(d2)