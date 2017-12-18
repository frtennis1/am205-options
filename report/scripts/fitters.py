from sklearn.preprocessing import PolynomialFeatures


def fd_fitter(C_grid, tol=0.):
    """Implements finite difference fitting"""
    dts = np.diff(C_grid.index)

    # Compute dCdt
    dCdt = np.clip((C_grid.as_matrix()[1:, :]
                    - C_grid.as_matrix()[:-1, :])
                   / dts[:, np.newaxis], a_min=0, a_max=None)

    # Centered difference dcdt
    dCdt_compatible = (dCdt[1:, 1:-1] + dCdt[:-1, 1:-1]) / 2

    # Assume equal strikes
    Kdiff = np.diff(C_grid.columns)[0]
    d2CdK2 = (C_grid.as_matrix()[:, :-2] + C_grid.as_matrix()[:, 2:]
              - 2 * C_grid.as_matrix()[:, 1:-1]) / Kdiff**2
    d2CdK2_compatible = d2CdK2[1:-1, :]

    # set second derivative of < tol to np.nan
    d2CdK2_compatible[d2CdK2_compatible < tol] = np.nan

    new_expiries = C_grid.index[1:-1]
    new_strikes = np.array(C_grid.columns[1:-1])

    local_vol2 = (dCdt_compatible / (1 / 2 * np.array(new_strikes **
                                                      2).reshape(1, -1) * d2CdK2_compatible))

    # convert to DataFrame and return
    return (pd.DataFrame(np.sqrt(local_vol2), index=new_expiries, columns=new_strikes),
            pd.DataFrame(dCdt_compatible, index=new_expiries,
                         columns=new_strikes),
            pd.DataFrame(d2CdK2_compatible, index=new_expiries, columns=new_strikes))


def local_quadratic_reg_fit(pxs, tol=0):
    """Implements local quadratic regression fitting"""
    def compute_derivatives(coeffs, strike, expiry):
        """Computes derivatives of a quadratic given coeff"""
        return (coeffs[2] + coeffs[4] * strike + 2 * coeffs[5] * expiry,
                2 * coeffs[3])

    # Initialize
    local_vol2 = np.zeros((len(pxs) - 2, len(pxs.columns) - 2))
    dCdTs = np.zeros((len(pxs) - 2, len(pxs.columns) - 2))
    d2CdK2s = np.zeros((len(pxs) - 2, len(pxs.columns) - 2))

    for i in range(1, len(pxs) - 1):
        for j in range(1, len(pxs.columns) - 1):

            # Stencil of point at position [i,j]
            local = pxs.iloc[i - 1:i + 2, j - 1:j + 2]

            # K, T of point [i,j]
            strike = pxs.columns[j]
            expiry = pxs.index[i]

            strikes, expiries = np.meshgrid(local.columns, local.index)

            # Transform K,T to 1, K, T, K^2, KT, T^2
            poly = PolynomialFeatures(degree=2)
            features = poly.fit_transform(
                np.vstack([strikes.flatten(), expiries.flatten()]).T)

            # Regression
            coeffs, _, _, _ = np.linalg.lstsq(
                features, local.as_matrix().flatten())

            dCdT, d2CdK2 = compute_derivatives(coeffs, strike, expiry)

            # Cleaning of resulting derivatives
            dCdT = np.clip(dCdT, a_min=0, a_max=None)
            d2CdK2 = d2CdK2 if d2CdK2 >= tol else np.nan

            local_vol2[i - 1, j - 1] = dCdT / (1 / 2 * (strike ** 2) * d2CdK2)
            dCdTs[i - 1, j - 1] = dCdT
            d2CdK2s[i - 1, j - 1] = d2CdK2

    # Construct DataFrames and returning
    local_vol = pd.DataFrame(np.sqrt(local_vol2), columns=pxs.columns[
                             1:-1], index=pxs.index[1:-1])
    dCdTs = pd.DataFrame(dCdTs, columns=pxs.columns[
                         1:-1], index=pxs.index[1:-1])
    d2CdK2s = pd.DataFrame(d2CdK2s, columns=pxs.columns[
                           1:-1], index=pxs.index[1:-1])
    return local_vol, dCdTs, d2CdK2s






