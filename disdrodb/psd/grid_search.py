# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Routines for grid search optimization."""

import numpy as np

DISTRIBUTION_TARGETS = {"N(D)", "H(x)"}
MOMENTS = {"M0", "M1", "M2", "M3", "M4", "M5", "M6"}
INTEGRAL_TARGETS = {"Z", "R", "LWC"} | MOMENTS
TARGETS = DISTRIBUTION_TARGETS | INTEGRAL_TARGETS

TRANSFORMATIONS = {"identity", "log", "sqrt"}
CENSORING = {"none", "left", "right", "both"}

DISTRIBUTION_METRICS = {"SSE", "SAE", "MAE", "MSE", "RMSE", "relMAE", "KLDiv", "WD", "JSD", "KS"}
INTEGRAL_METRICS = {"SE", "AE"}
ERROR_METRICS = DISTRIBUTION_METRICS | INTEGRAL_METRICS


def check_target(target):
    """Check valid target argument."""
    valid_targets = TARGETS
    if target not in valid_targets:
        raise ValueError(f"Invalid 'target' {target}. Valid targets are {valid_targets}.")
    return target


def check_censoring(censoring):
    """Check valid censoring argument."""
    valid_censoring = CENSORING
    if censoring not in valid_censoring:
        raise ValueError(f"Invalid 'censoring' {censoring}. Valid options are {valid_censoring}.")
    return censoring


def check_transformation(transformation):
    """Check valid transformation argument."""
    valid_transformation = TRANSFORMATIONS
    if transformation not in valid_transformation:
        raise ValueError(
            f"Invalid 'transformation' {transformation}. Valid options are {valid_transformation}.",
        )
    return transformation


def check_loss(loss, valid_metrics=ERROR_METRICS):
    """Check valid loss argument."""
    if loss not in valid_metrics:
        raise ValueError(f"Invalid 'loss' {loss}. Valid options are {valid_metrics}.")
    return loss


def check_valid_loss(loss, target):
    """Check if loss is valid for the given target.

    For distribution targets (ND, H(x)), any error metric is valid.
    For scalar targets (Z, R, LWC), distribution error metrics are not valid.

    Parameters
    ----------
    loss : str
        The error metric to validate.
    target : str
        The target variable type.

    Returns
    -------
    str
        The validated loss.

    Raises
    ------
    ValueError
        If loss is not valid for the given target.
    """
    if target in {"N(D)", "H(x)"}:
        return check_loss(loss, valid_metrics=DISTRIBUTION_METRICS)
    # Integral N(D) target (Z, R, LWC, M1, ..., M6)
    return check_loss(loss, valid_metrics=INTEGRAL_METRICS)


def check_loss_weight(loss_weight):
    """Check valid loss_weight argument."""
    if loss_weight <= 0:
        raise ValueError(f"Invalid 'loss_weight' {loss_weight}. Must be greater than 0.")
    return loss_weight


def check_objective(objective):
    """Check objective validity."""
    # Check required keys are present
    required_keys = {"target", "transformation", "censoring", "loss"}
    missing_keys = required_keys - set(objective.keys())
    if missing_keys:
        raise ValueError(
            f"Objective {objective} is missing required keys: {missing_keys}. " f"Required keys are: {required_keys}",
        )

    # Validate target
    objective["target"] = check_target(objective["target"])

    # Validate transformation
    objective["transformation"] = check_transformation(objective["transformation"])

    # Validate censoring
    objective["censoring"] = check_censoring(objective["censoring"])

    # Validate loss and check compatibility with target
    objective["loss"] = check_loss(objective["loss"])
    objective["loss"] = check_valid_loss(objective["loss"], objective["target"])
    return objective


def check_objectives(objectives):
    """Validate and normalize objectives for grid search optimization.

    Parameters
    ----------
    objectives : list of dicts
        List of objective dictionaries, each containing:
        - 'target' : str, Target variable (N(D), H(x), R, Z, LWC, or M<p>)
        - 'transformation' : str, Transformation type (identity, log, sqrt)
        - 'censoring' : str, Censoring type (none, left, right, both)
        - 'loss' : str, Error metric (SSE, SAE, MAE, MSE, RMSE, etc.)
        - 'loss_weight' : float, optional, Weight for weighted optimization (auto-set to 1.0 for single objective)

    Returns
    -------
    list of dicts
        Validated objectives. Loss weights are not normalized.
    """
    if objectives is None:
        return None
    if not isinstance(objectives, list) or len(objectives) == 0:
        raise TypeError("'objectives' must be a non-empty list of dictionaries.")

    if not all(isinstance(obj, dict) for obj in objectives):
        raise TypeError("All items in 'objectives' must be dictionaries.")

    # Validate each objective
    for idx, objective in enumerate(objectives):
        objectives[idx] = check_objective(objective)

    # Handle loss_weight
    if len(objectives) == 1:
        # Single objective: auto-set weight to 1.0 if not provided
        if "loss_weight" in objectives[0]:
            raise ValueError(
                "'loss_weight' should be specified only if multiple objectives are used.",
            )
    else:
        # Multiple objectives: verify all have weights and normalize
        for objective in objectives:
            if "loss_weight" not in objective:
                raise ValueError(
                    f"Objective {objective} is missing 'loss_weight'. "
                    f"When using multiple objectives, all must have 'loss_weight' specified.",
                )
            objective["loss_weight"] = check_loss_weight(objective["loss_weight"])
    return objectives


####---------------------------------------------------------------------------
#### Targets


def compute_rain_rate(ND, D, dD, V):
    """Compute rain rate from drop size distribution.

    Parameters
    ----------
    ND : np.ndarray
        Drop size distribution [#/m3/mm-1]. Can be 1D [n_bins] or 2D [n_samples, n_bins].
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins] [m/s]

    Returns
    -------
    np.ndarray
        Rain rate [mm/h]
    """
    axis = 1 if ND.ndim == 2 else None
    rain_rate = np.pi / 6 * np.sum(ND * V * (D / 1000) ** 3 * dD, axis=axis) * 3600 * 1000
    return rain_rate  # mm/h


def compute_lwc(ND, D, dD, rho_w=1000):
    """Compute liquid water content from drop size distribution.

    Parameters
    ----------
    ND : np.ndarray
        Drop size distribution [#/m3/mm-1]. Can be 1D [n_bins] or 2D [n_samples, n_bins].
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]
    rho_w : float, optional
        Water density [kg/m3]. Default is 1000.

    Returns
    -------
    np.ndarray
        Liquid water content [g/m3]
    """
    axis = 1 if ND.ndim == 2 else None
    lwc = np.pi / 6.0 * (rho_w * 1000) * np.sum((D / 1000) ** 3 * ND * dD, axis=axis)
    return lwc  # g/m3


def compute_moment(ND, order, D, dD):
    """Compute moment of the drop size distribution.

    Parameters
    ----------
    ND : np.ndarray
        Drop size distribution [#/m3/mm-1]. Can be 1D [n_bins] or 2D [n_samples, n_bins].
    order : int
        Moment order.
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]

    Returns
    -------
    np.ndarray
        Moment of the specified order [mm^order·m^-3]
    """
    axis = 1 if ND.ndim == 2 else None
    return np.sum((D**order * ND * dD), axis=axis)  # mm**order·m⁻³


def compute_z(ND, D, dD):
    """Compute radar reflectivity from drop size distribution.

    Parameters
    ----------
    ND : np.ndarray
        Drop size distribution [#/m3/mm-1]. Can be 1D [n_bins] or 2D [n_samples, n_bins].
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]

    Returns
    -------
    np.ndarray
        Radar reflectivity [dBZ]
    """
    z = compute_moment(ND, order=6, D=D, dD=dD)  # mm⁶·m⁻³
    Z = 10 * np.log10(np.where(z > 0, z, np.nan))
    return Z


def compute_target_variable(
    target,
    ND_obs,
    ND_preds,
    D,
    dD,
    V,
):
    """Compute target variable from drop size distribution.

    Parameters
    ----------
    target : str
        Target variable type. Can be 'Z', 'R', 'LWC', moments ('M0'-'M6'), 'N(D)', or 'H(x)'.
    ND_obs : 1D array
        Observed drop size distribution [#/m3/mm-1] [n_bins]
    ND_preds : 2D array
        Predicted drop size distributions [n_samples, n_bins] [#/m3/mm-1]
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins] [m/s]

    Returns
    -------
    tuple
        (obs, pred) where obs is 1D [n_bins] or scalar, and pred is 2D [n_samples, n_bins] or 1D [n_samples]
    """
    # Compute observed and predicted target variables
    if target == "Z":
        obs = compute_z(ND_obs, D=D, dD=dD)
        pred = compute_z(ND_preds, D=D, dD=dD)
    elif target == "R":
        obs = compute_rain_rate(ND_obs, D=D, dD=dD, V=V)
        pred = compute_rain_rate(ND_preds, D=D, dD=dD, V=V)
    elif target == "LWC":
        obs = compute_lwc(ND_obs, D=D, dD=dD)
        pred = compute_lwc(ND_preds, D=D, dD=dD)
    elif target in MOMENTS:
        order = int(target[1])
        obs = compute_moment(ND_obs, order=order, D=D, dD=dD)
        pred = compute_moment(ND_preds, order=order, D=D, dD=dD)
    else:  # N(D) or H(x)
        obs = ND_obs
        pred = ND_preds
    return obs, pred


####---------------------------------------------------------------------------
#### Censoring


def left_truncate_bins(ND_obs, ND_preds, D, dD, V):
    """Truncate left side of bins (smallest diameters) to first non-zero bin.

    Parameters
    ----------
    ND_obs : 1D array
        Observed drop size distribution [n_bins]
    ND_preds : 2D array
        Predicted drop size distributions [n_samples, n_bins]
    D : 1D array
        Diameter bin center in mm [n_bins]
    dD : 1D array
        Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins]

    Returns
    -------
    tuple or None
        (ND_obs_trunc, ND_preds_trunc, D_trunc, dD_trunc, V_trunc) or None if all zeros
    """
    if np.all(ND_obs == 0):  # all zeros
        return None
    idx = np.argmax(ND_obs > 0)
    return (
        ND_obs[idx:],
        ND_preds[:, idx:],
        D[idx:],
        dD[idx:],
        V[idx:],
    )


def right_truncate_bins(ND_obs, ND_preds, D, dD, V):
    """Truncate right side of bins (largest diameters) to last non-zero bin.

    Parameters
    ----------
    ND_obs : 1D array
        Observed drop size distribution [n_bins]
    ND_preds : 2D array
        Predicted drop size distributions [n_samples, n_bins]
    D : 1D array
        Diameter bin center in mm [n_bins]
    dD : 1D array
        Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins]

    Returns
    -------
    tuple or None
        (ND_obs_trunc, ND_preds_trunc, D_trunc, dD_trunc, V_trunc) or None if all zeros
    """
    if np.all(ND_obs == 0):  # all zeros
        return None
    idx = len(ND_obs) - np.argmax(ND_obs[::-1] > 0)
    return (
        ND_obs[:idx],
        ND_preds[:, :idx],
        D[:idx],
        dD[:idx],
        V[:idx],
    )


def truncate_bin_edges(
    ND_obs,
    ND_preds,
    D,
    dD,
    V,
    left_censored=False,
    right_censored=False,
):
    """Truncate bin edges based on censoring strategy.

    Parameters
    ----------
    ND_obs : 1D array
        Observed drop size distribution [n_bins]
    ND_preds : 2D array
        Predicted drop size distributions [n_samples, n_bins]
    D : 1D array
        Diameter bin center in mm [n_bins]
    dD : 1D array
        Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins]
    left_censored : bool, optional
        If True, truncate from the left (remove small diameter bins). Default is False.
    right_censored : bool, optional
        If True, truncate from the right (remove large diameter bins). Default is False.

    Returns
    -------
    tuple or None
        (ND_obs_trunc, ND_preds_trunc, D_trunc, dD_trunc, V_trunc) or None if all zeros
    """
    data = (ND_obs, ND_preds, D, dD, V)
    if left_censored:
        data = left_truncate_bins(*data)
        if data is None:
            return None
    if right_censored:
        data = right_truncate_bins(*data)
        if data is None:
            return None
    return data


####---------------------------------------------------------------------------
#### Transformation


def apply_transformation(obs, pred, transformation):
    """Apply transformation to observed and predicted values.

    Parameters
    ----------
    obs : np.ndarray
        Observed values
    pred : np.ndarray
        Predicted values
    transformation : str
        Transformation type: 'identity', 'log', or 'sqrt'.

    Returns
    -------
    tuple
        (obs_transformed, pred_transformed)
    """
    if transformation == "log":
        return np.log(obs + 1), np.log(pred + 1)
    if transformation == "sqrt":
        return np.sqrt(obs), np.sqrt(pred)
    # if transformation == "identity":
    return obs, pred


####---------------------------------------------------------------------------
#### Loss  metrics
def _compute_kl(p_k, q_k, eps=1e-12):
    """Compute Kullback-Leibler divergence.

    Parameters
    ----------
    p_k : np.ndarray
        Reference probability distribution [n_samples, n_bins] or [1, n_bins]
    q_k : np.ndarray
        Comparison probability distribution [n_samples, n_bins]
    eps : float, optional
        Small value for numerical stability. Default is 1e-12.

    Returns
    -------
    np.ndarray
        KL divergence for each sample [n_samples]
    """
    q_safe = np.maximum(q_k, eps)
    kl = np.sum(
        p_k * np.log(p_k / q_safe),
        axis=1,
        where=(p_k > 0),  # sum where p > 0
    )
    # Clip to 0
    kl = np.maximum(kl, 0.0)

    # Set to NaN if probability mass is all 0
    pk_mass = p_k.sum()
    qk_mass = q_k.sum(axis=1)
    kl = np.where(pk_mass > 0, kl, np.nan)
    kl = np.where(qk_mass > 0, kl, np.nan)
    return kl


def compute_kl_divergence(obs, pred, dD, eps=1e-12):
    """Compute Kullback-Leibler divergence between observed and predicted N(D).

    Parameters
    ----------
    obs : 1D array
        Observed N(D) values [n_bins]. Unit [#/m3/mm-1]
    pred : 2D array
        Predicted N(D) values [n_samples, n_bins]. Unit [#/m3/mm-1]
    dD : 1D array
       Diameter bin width in mm [n_bins]

    Returns
    -------
    np.ndarray
        KL divergence for each sample [n_samples]
    """
    # Convert N(D) to probabilities (normalize by bin width and total)
    # pdf =  N(D) * dD / sum( N(D) * dD)
    p_k = (obs * dD) / (np.sum(obs * dD) + eps)
    q_k = (pred * dD[None, :]) / (np.sum(pred * dD[None, :], axis=1, keepdims=True) + eps)

    # KL(P||Q) = sum(P * log(P/Q))
    kl = _compute_kl(p_k=p_k[None, :], q_k=q_k, eps=eps)
    return kl


def compute_jensen_shannon_distance(obs, pred, dD, eps=1e-12):
    """Compute Jensen-Shannon distance between observed and predicted N(D).

    The Jensen-Shannon distance is the square root of the Jensen-Shannon divergence.
    Values are defined between 0 and np.sqrt(ln(2)) = 0.83256

    Vectorized implementation for multiple predictions.

    Parameters
    ----------
    obs : 1D array
        Observed N(D) values [n_bins]. Unit [#/m3/mm-1]
    pred : 2D array
        Predicted N(D) values [n_samples, n_bins]. Unit [#/m3/mm-1]
    dD : 1D array
       Diameter bin width in mm [n_bins]

    Returns
    -------
    np.ndarray
        Jensen-Shannon distance for each sample [n_samples]
    """
    # Convert N(D) to probability distributions
    obs_prob = (obs * dD) / (np.sum(obs * dD) + eps)
    pred_prob = (pred * dD[None, :]) / (np.sum(pred * dD[None, :], axis=1, keepdims=True) + eps)

    # Mixture distribution
    M = 0.5 * (obs_prob[None, :] + pred_prob)

    # Compute KL divergences
    # - KL(P||M)
    kl_obs = _compute_kl(p_k=obs_prob[None, :], q_k=M, eps=eps)

    # - KL(Q||M)
    kl_pred = _compute_kl(p_k=pred_prob, q_k=M, eps=eps)

    # Compute Jensen Shannon divergence
    js_div = 0.5 * (kl_obs + kl_pred)
    js_div = np.maximum(js_div, 0.0)  # clip tiny negative values to zero (numerical safety)

    # Jensen-Shannon distance
    js_distance = np.sqrt(js_div)
    js_distance = np.maximum(js_distance, 0.0)
    return js_distance


def compute_wasserstein_distance(obs, pred, D, dD, eps=1e-12, integration="bin"):
    """Compute Wasserstein distance (Earth Mover's Distance) between observed and predicted N(D).

    Vectorized implementation for multiple predictions.

    Parameters
    ----------
    obs : 1D array
        Observed N(D) values [n_bins]. Unit [#/m3/mm-1]
    pred : 2D array
        Predicted N(D) values [n_samples, n_bins]. Unit [#/m3/mm-1]
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]
    integration : {"bin", "left_riemann"}
        Integration scheme for the Wasserstein integral:

        - "bin":
            Histogram-based Wasserstein distance.
            Interprets N(D) as a piecewise-constant density over bins of width dD
            and integrates the CDF difference over those intervals.
            Assumes the CDF difference is constant within each bin and
            integrates using the bin widths (dD).

        - "left_riemann":
            Discrete-support Wasserstein distance.
            Interprets probability mass as concentrated at bin centers D and
            integrates using spacing between support points, consistent with
            scipy.stats.wasserstein_distance.
            Use Left Riemann sum using bin centers.

    Returns
    -------
    np.ndarray
        Wasserstein distance for each sample [n_samples]
    """
    # from scipy.stats import wasserstein_distance

    # wasserstein_distance(
    #     u_values=D,
    #     v_values=D,
    #     u_weights=obs_prob,
    #     v_weights=pred_prob[0]
    # )

    # Convert N(D) to probabilities (normalize by bin width and total)
    # pdf =  N(D) * dD / sum( N(D) * dD)
    obs_prob = (obs * dD) / (np.sum(obs * dD) + eps)
    pred_prob = (pred * dD[None, :]) / (np.sum(pred * dD[None, :], axis=1, keepdims=True) + eps)

    # Compute cumulative distributions
    obs_cdf = np.cumsum(obs_prob)
    pred_cdf = np.cumsum(pred_prob, axis=1)

    # Wasserstein distance = integral of |CDF_obs - CDF_pred| over D
    # - Compute difference between CDFs
    obs_cdf_expanded = obs_cdf[None, :]  # [1, n_bins]
    diff = np.abs(obs_cdf_expanded - pred_cdf)  # [n_samples, n_bins]

    if integration == "bin":
        wd = np.sum(diff * dD[None, :], axis=1)
    else:
        # Integrate using left Riemann sum (as Scipy wasserstein_distance)
        dx = np.diff(D)
        wd = np.sum(diff[:, :-1] * dx[None, :], axis=1)

    # Clip to 0
    wd = np.maximum(wd, 0.0)

    # Set to NaN if probability mass is all 0
    obs_mass = obs_prob.sum()
    pred_mass = pred_prob.sum(axis=1)
    wd = np.where(obs_mass > 0, wd, np.nan)
    wd = np.where(pred_mass > 0, wd, np.nan)
    return wd


def compute_kolmogorov_smirnov_distance(obs, pred, dD, eps=1e-12):
    """Compute Kolmogorov-Smirnov (KS) distance between observed and predicted N(D).

    The Kolmogorov-Smirnov (KS) distance is bounded between 0 and 1,
    where 0 indicates that the two distributions are identical.
    The associated KS test p-value ranges from 0 to 1,
    with a value of 1 indicating no evidence against the null hypothesis that the distributions are identical.
    When the p value is smaller than the significance level (e.g. < 0.05) the model is rejected.

    If model parameters are estimated from the same data to which the model is compared,
    the standard KS p-values are invalid.
    The solution is to use a parametric bootstrap:
    1. Fit model to your data
    2. Simulate many datasets from that fitted gamma
    3. Refit gamma for each simulated dataset
    4. Compute KS statistic each time
    5. Compare your observed KS statistic to the bootstrap distribution

    Vectorized implementation for multiple predictions.

    Parameters
    ----------
    obs : 1D array
        Observed N(D) values [n_bins]. Unit [#/m3/mm-1]
    pred : 2D array
        Predicted N(D) values [n_samples, n_bins]. Unit [#/m3/mm-1]
    dD : 1D array
       Diameter bin width in mm [n_bins]

    Returns
    -------
    np.ndarray
        KS statistic for each sample [n_samples]
        If 0, the two distributions are identical.
    np.ndarray
        KS p-value for each sample [n_samples]
        A p-value of 0 means “strong evidence against equality.”
        A p-value of 1 means “no evidence against equality.”
        Identical distributions show a pvalue of 1.
        Similar distributions show a pvalue close to 1.
    """
    # Convert N(D) to probability mass
    obs_prob = (obs * dD) / (np.sum(obs * dD) + eps)
    pred_prob = (pred * dD[None, :]) / (np.sum(pred * dD[None, :], axis=1, keepdims=True) + eps)

    # Compute CDFs
    obs_cdf = np.cumsum(obs_prob)  # (n_bins,)
    pred_cdf = np.cumsum(pred_prob, axis=1)  # (n_samples, n_bins)

    # KS statistic = max |CDF_obs - CDF_pred|
    ks = np.max(np.abs(pred_cdf - obs_cdf[None, :]), axis=1)

    # Compute effective sample sizes (from probabilities)
    n_eff_obs = 1.0 / np.sum(obs_prob**2)
    n_eff_pred = 1.0 / np.sum(pred_prob**2, axis=1)
    n_eff_ks = (n_eff_obs * n_eff_pred) / (n_eff_obs + n_eff_pred)

    # Compute KS pvalue (asymptotic approximation)
    p_value = 2.0 * np.exp(-2.0 * (ks * np.sqrt(n_eff_ks)) ** 2)
    p_value = np.clip(p_value, 0.0, 1.0)

    # Set to NaN if probability mass is all 0
    obs_mass = obs_prob.sum()
    pred_mass = pred_prob.sum(axis=1)
    ks = np.where(obs_mass > 0, ks, np.nan)
    ks = np.where(pred_mass > 0, ks, np.nan)
    p_value = np.where(obs_mass > 0, p_value, np.nan)
    p_value = np.where(pred_mass > 0, p_value, np.nan)
    return ks, p_value


####---------------------------------------------------------------------------
#### Wrappers


def compute_errors(obs, pred, loss, D=None, dD=None):  # noqa: PLR0911
    """Compute error between observed and predicted values.

    The function is entirely vectorized and can handle multiple predictions at once.

    Parameters
    ----------
    obs : np.ndarray
        Observed values.
        Is scalar value if specified target is an integral variable.
        Is 1D array of size [n_bins] if target is a distribution.
    pred : np.ndarray
        Predicted values. Can be 1D [n_samples] or 2D [n_samples, n_bins].
        Is 1D when specified target is an integral variable.
        Is 2D when specified target is a distribution.
    loss : str
        Error metric to compute. See supported metrics in ERROR_METRICS.
    D : 1D array, optional
        Diameter bin center in mm [n_bins]. Required for 'WD' metric. Default is None.
    dD : 1D array, optional
        Diameter bin width in mm [n_bins]. Required for distribution metrics. Default is None.

    Returns
    -------
    np.ndarray
        Computed error(s) [n_samples] for most metrics, or [n_samples, n_bins] for element-wise metrics.
    """
    # Handle scalar obs case (from integral targets like Z, R, LWC)
    if np.isscalar(obs):
        obs = np.asarray(obs)

    # Compute SE or AE (for integral targets)
    if obs.size == 1:
        if loss == "AE":
            return np.abs(obs - pred)
        # "SE"
        return (obs - pred) ** 2

    # Compute KL or WD if asked (obs is expanded internally to save computations)
    if loss == "KLDiv":
        return compute_kl_divergence(obs, pred, dD=dD)
    if loss == "JSD":
        return compute_jensen_shannon_distance(obs, pred, dD=dD)
    if loss == "WD":
        return compute_wasserstein_distance(obs, pred, D=D, dD=dD)
    if loss == "KS":
        return compute_kolmogorov_smirnov_distance(obs, pred, dD=dD)[0]  # select distance
    # if loss == "KS_pvalue":
    #     return compute_kolmogorov_smirnov_distance(obs, pred, dD=dD)[1]  # select p_value

    # Broadcast obs to match pred shape if needed (when target is N(D) or H(x))
    # If obs is 1D and pred is 2D, add dimension to obs
    if pred.ndim > obs.ndim:
        obs = obs[None, :]

    # Compute error metrics
    if loss == "SSE":
        return np.sum((obs - pred) ** 2, axis=1)
    if loss == "SAE":
        return np.sum(np.abs(obs - pred), axis=1)
    if loss == "MAE":
        return np.mean(np.abs(obs - pred), axis=1)
    if loss == "relMAE":
        return np.mean(np.abs(obs - pred) / (np.abs(obs) + 1e-12), axis=1)
    if loss == "MSE":
        return np.mean((obs - pred) ** 2, axis=1)
    if loss == "RMSE":
        return np.sqrt(np.mean((obs - pred) ** 2, axis=1))
    raise NotImplementedError(f"Error metric '{loss}' is not implemented.")


def normalize_errors(errors):
    """Normalize errors to scale minimum error region to O(1).

    Scaling by the median value of the p0-p10 region normalizes error in
    the minimum region to approximately O(1). Scaling by p95-p5 is not used
    because when tails span orders of magnitude, it normalizes the spread
    rather than the minimum region, suppressing the minimum region and
    amplifying the bad region.

    Parameters
    ----------
    errors : np.ndarray
        Error values to normalize [n_samples]

    Returns
    -------
    np.ndarray
        Normalized errors (if normalize_error=True) or original errors (if False)
    """
    p10 = np.nanpercentile(errors, q=10)
    scale = np.nanmedian(errors[errors <= p10])

    ## Investigate normalization
    # plt.hist(errors[errors < p10], bins=100)
    # errors_norm = errors / scale
    # p_norm10 = np.nanpercentile(errors_norm, q=10)
    # plt.hist(errors_norm[errors_norm < p_norm10], bins=100)

    # scale = np.diff(np.nanpercentile(errors, q=[1, 99])) + 1e-12
    if scale != 0:
        errors = errors / scale
    return errors


def compute_loss(
    ND_obs,
    ND_preds,
    D,
    dD,
    V,
    target,
    censoring,
    transformation,
    loss,
    check_arguments=True,
):
    """Compute loss.

    Computes loss between observed and predicted drop size distributions,
    with optional censoring, transformation, and target variable specification.

    Parameters
    ----------
    ND_obs : 1D array
        Observed drop size distribution [n_bins] [#/m3/mm-1]
    ND_preds : 2D array
        Predicted drop size distributions [n_samples, n_bins] [#/m3/mm-1]
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins] [m/s]
    target : str
        Target variable: 'Z', 'R', 'LWC', moments ('M0'-'M6'), 'N(D)', or 'H(x)'.
    censoring : str
        Censoring strategy: 'none', 'left', 'right', or 'both'.
    transformation : str
        Transformation: 'identity', 'log', or 'sqrt'.
    loss : str
        Loss function.
        If target is ``"N(D)"`` or ``"H(x)"``, valid options are:
        - ``SSE``: Sum of Squared Errors
        - ``SAE``: Sum of Absolute Errors
        - ``MAE``: Mean Absolute Error
        - ``MSE``: Mean Squared Error
        - ``RMSE``: Root Mean Squared Error
        - ``relMAE``: Relative Mean Absolute Error
        - ``KLDiv``: Kullback-Leibler Divergence
        - ``WD``: Wasserstein Distance
        - ``JSD``: Jensen-Shannon Distance
        - ``KS``: Kolmogorov-Smirnov Statistic
        If target is one of ``"R"``, ``"Z"``, ``"LWC"``, or ``"M<p>"``, valid options are:
        - ``AE``: Absolute Error
        - ``SE``: Squared Error
    check_arguments : bool, optional
        If True, validate input arguments. Default is True.

    Returns
    -------
    np.ndarray
        Computed errors [n_samples]. Values are NaN where computation failed.
    """
    # Check input
    if check_arguments:
        target = check_target(target)
        transformation = check_transformation(transformation)
        censoring = check_censoring(censoring)
        loss = check_valid_loss(loss, target=target)

    # Clip N(D) < 1e-3 to 0
    ND_obs = np.where(ND_obs < 1e-3, 0.0, ND_obs)
    ND_preds = np.where(ND_preds < 1e-3, 0.0, ND_preds)

    # Truncate if asked
    left_censored = censoring in {"left", "both"}
    right_censored = censoring in {"right", "both"}
    if left_censored or right_censored:
        truncated = truncate_bin_edges(
            ND_obs,
            ND_preds,
            D,
            dD,
            V,
            left_censored=left_censored,
            right_censored=right_censored,
        )
        if truncated is None:
            # Grid search logic expects inf so it can be turned into NaN later
            return np.full(ND_preds.shape[0], np.inf)
        ND_obs, ND_preds, D, dD, V = truncated

    # Compute target variable
    obs, pred = compute_target_variable(target, ND_obs, ND_preds, D=D, dD=dD, V=V)

    # Apply transformation
    obs, pred = apply_transformation(obs, pred, transformation=transformation)

    # Compute errors
    errors = compute_errors(obs, pred, loss=loss, D=D, dD=dD)

    # Replace inf with NaN
    errors[~np.isfinite(errors)] = np.nan
    return errors


def compute_weighted_loss(ND_obs, ND_preds, D, dD, V, objectives, Nc=None):
    """Compute weighted loss between observed and predicted particle size distributions.

    Parameters
    ----------
    ND_obs : 1D array
        Observed drop size distribution [n_bins] [#/m3/mm-1]
    ND_preds : 2D array
        Predicted drop size distributions [n_samples, n_bins] [#/m3/mm-1]
    D : 1D array
        Diameter bin centers in mm [n_bins]
    dD : 1D array
       Diameter bin width in mm [n_bins]
    V : 1D array
        Terminal velocity [n_bins] [m/s]
    objectives: list of dicts
        target : str, optional
            Target quantity to optimize. Valid options:
            - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹]
            - ``"H(x)"`` : Normalized drop number concentration [-]
            - ``"R"`` : Rain rate [mm h⁻¹]
            - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
            - ``"LWC"`` : Liquid water content [g m⁻³]
            - ``"M<p>"`` : Moment of order p
        transformation : str, optional
            Transformation applied to the target quantity before computing the loss.
            Valid options:
            - ``"identity"`` : No transformation
            - ``"log"`` : Logarithmic transformation
            - ``"sqrt"`` : Square root transformation
        censoring : str
            Specifies whether the observed particle size distribution (PSD) is
            treated as censored at the edges of the diameter range due to
            instrumental sensitivity limits:
            - ``"none"`` : No censoring is applied. All diameter bins are used.
            - ``"left"`` : Left-censored PSD. Diameter bins at the lower end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"right"`` : Right-censored PSD. Diameter bins at the upper end of
              the spectrum where the observed number concentration is zero are
              removed prior to cost-function evaluation.
            - ``"both"`` : Both left- and right-censored PSD. Only the contiguous
              range of diameter bins with non-zero observed concentrations is
              retained.
        loss : int, optional
            Loss function.
            If target is ``"N(D)"`` or ``"H(x)"``, valid options are:
            - ``SSE``: Sum of Squared Errors
            - ``SAE``: Sum of Absolute Errors
            - ``MAE``: Mean Absolute Error
            - ``MSE``: Mean Squared Error
            - ``RMSE``: Root Mean Squared Error
            - ``relMAE``: Relative Mean Absolute Error
            - ``KLDiv``: Kullback-Leibler Divergence
            - ``WD``: Wasserstein Distance
            - ``JSD``: Jensen-Shannon Distance
            - ``KS``: Kolmogorov-Smirnov Statistic
            If target is one of ``"R"``, ``"Z"``, ``"LWC"``, or ``"M<p>"``, valid options are:
            - ``AE``: Absolute Error
            - ``SE``: Squared Error
        loss_weight: int, optional
            Weight of this objective when multiple objectives are used.
            Must be specified if len(objectives) > 1.
    Nc : float, optional
        Normalization constant for H(x) target.
        If provided, N(D) will be divided by Nc.

    Returns
    -------
    np.ndarray
        Computed errors [n_samples]. Values are NaN where computation failed.
    """
    # Compute weighted loss across all targets
    total_loss = np.zeros(ND_preds.shape[0])
    total_loss_weights = 0
    for objective in objectives:
        # Extract target configuration
        target = objective["target"]
        loss = objective.get("loss", None)
        censoring = objective["censoring"]
        transformation = objective["transformation"]
        if len(objectives) > 1:
            loss_weight = objective["loss_weight"]
            normalize_loss = True  # objective["normalize_loss"]
        else:
            loss_weight = 1
            normalize_loss = False  # objective["normalize_loss"]

        # Prepare observed and predicted variables
        # - Compute normalized H(x) if Nc provided and target is H(x)
        if Nc is not None:
            obs = ND_obs / Nc if target == "H(x)" else ND_obs
            preds = ND_preds / Nc if target == "H(x)" else ND_preds
        else:
            obs = ND_obs
            preds = ND_preds

        # Compute errors for this target
        loss_values = compute_loss(
            ND_obs=obs,
            ND_preds=preds,
            D=D,
            dD=dD,
            V=V,
            target=target,
            transformation=transformation,
            loss=loss,
            censoring=censoring,
        )

        # Normalize loss
        if normalize_loss:
            loss_values = normalize_errors(loss_values)

        # Accumulate weighted loss
        total_loss += loss_weight * loss_values
        total_loss_weights += loss_weight

    # Normalize by total weight
    total_loss = total_loss / total_loss_weights

    # Replace inf with NaN
    total_loss[~np.isfinite(total_loss)] = np.nan
    return total_loss
