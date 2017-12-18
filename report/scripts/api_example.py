# Example of using LocalVolatilityPricer
expiries = np.arange(1 / 12, 13 / 12, 1 / 12)
K = np.arange(50, 151, 1)


def quadratic_vol(px, time):
    return np.clip(.16 + 1e-4 * (px - 100) ** 2, 0, .5)

# Instantiate object
qv = LocalVolatilityPricer(quadratic_vol, expiries, K)
_ = qv.get_true_prices(n_samples=int(1e5))

_ = qv.fit_local_vol(fd_fitter, interp_option='nn', tol=0.01)

# Check fitting at t = 0
qv.plot_fit(0)

# Check errors
_ = qv.generate_prediction(n_samples=int(1e5))
qv.plot_errors()
