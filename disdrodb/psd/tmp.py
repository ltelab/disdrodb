def get_normalized_generalized_gamma_parameters_gs(
    ds,
    psd_model_kwargs,
    target="N(D)",
    transformation="log",
    error_metric="MAE",
    censoring="none",
):
    """Estimate Normalized Generalized Gamma PSD parameters using Grid Search optimization.

    The parameters ``N_c`` and ``Dc`` are computed empirically from the observed DSD
    moments, while the shape parameters ``mu`` and ``c`` are estimated through
    grid search by minimizing the error between observed and modeled quantities.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PSD observations. Must include:
        - ``drop_number_concentration`` : Drop number concentration [m⁻³ mm⁻¹]
        - ``diameter_bin_center`` : Diameter bin centers [mm]
        - ``diameter_bin_width`` : Diameter bin widths [mm]
        - ``fall_velocity`` : Drop fall velocity [m s⁻¹] (required if target='R')
    psd_model_kwargs : dict 
        Dictionary with the i and j moment order to use.
    target : str, optional
        Target quantity to optimize. Valid options:
        - ``"N(D)"`` : Drop number concentration [m⁻³ mm⁻¹] (default)
        - ``"R"`` : Rain rate [mm h⁻¹]
        - ``"Z"`` : Radar reflectivity [mm⁶ m⁻³]
        - ``"LWC"`` : Liquid water content [g m⁻³]
    transformation : str, optional
        Transformation applied to the target quantity before computing the error.
        Valid options:
        - ``"identity"`` : No transformation
        - ``"log"`` : Logarithmic transformation (default)
        - ``"sqrt"`` : Square root transformation
    error_order : int, optional
        Order of the error metric (p-norm):
        - ``1`` : L1 norm / Mean Absolute Error (MAE) (default)
        - ``2`` : L2 norm / Mean Squared Error (MSE)
        Higher orders tend to emphasize larger errors and may stretch the
        fitted distribution toward higher diameters.
    censoring : {"none", "left", "right", "both"}, optional
        Specifies whether the observed particle size distribution (PSD) is
        treated as censored at the edges of the diameter range due to
        instrumental sensitivity limits.

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

    Returns
    -------
    ds_params : xarray.Dataset
        Dataset containing the estimated Normalized Generalized Gamma distribution parameters.
    """
    # Retrieve moments 
    i = psd_model_kwargs["i"]
    j = psd_model_kwargs["j"]

    # Compute required variables
    drop_number_concentration = ds["drop_number_concentration"]
    diameter_bin_width = ds["diameter_bin_width"]
    diameter = ds["diameter_bin_center"] / 1000  # conversion from mm to m    
    Mi = get_moment(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
        moment=i,
    )
    Mj = get_moment(
        drop_number_concentration=drop_number_concentration,
        diameter=diameter,  # m
        diameter_bin_width=diameter_bin_width,  # mm
        moment=j,
    )
    Dc = NormalizedGeneralizedGammaPSD.compute_Dc(i=i, j=j, Mi=Mi, Mj=Mj)
    Nc = NormalizedGeneralizedGammaPSD.compute_Nc(i=i, j=j, Mi=Mi, Mj=Mj)

    # Define kwargs
    kwargs = {
        "D": ds["diameter_bin_center"].data,
        "dD": ds["diameter_bin_width"].data,
        "target": target,
        "transformation": transformation,
        "error_metric": error_metric,
        "censoring": censoring,
    }

    # Fit distribution in parallel
    da_params = xr.apply_ufunc(
        apply_normalized_generalized_gamma_gs,
        # Variables varying over time
        i, 
        j, 
        Nc, 
        Dc, 
        ds["drop_number_concentration"],
        ds["fall_velocity"],
        # Other options
        kwargs=kwargs,
        # Settings
        input_core_dims=[[], [], [],[], [DIAMETER_DIMENSION], [DIAMETER_DIMENSION]],
        output_core_dims=[["parameters"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"parameters": 4}},  # lengths of the new output_core_dims dimensions.
        output_dtypes=["float64"],
    )

    # Add parameters coordinates
    da_params = da_params.assign_coords({"parameters": ["Nc", "Dc", "mu", "c"]})
 
    # Create parameters dataset
    ds_params = da_params.to_dataset(dim="parameters")
    
    # Add Nc and Dc 
    ds_params["Dc"].attrs["moment_orders"] = f"{i}, {j}"
    ds_params["Nc"].attrs["moment_orders"] = f"{i}, {j}"
    
    # Add DSD model name to the attribute
    ds_params.attrs["disdrodb_psd_model"] = "NormalizedGeneralizedGammaPSD"
    ds_params.attrs["disdrodb_psd_model_kwargs"] = str(psd_model_kwargs)
    return ds_params


def apply_normalized_generalized_gamma_gs(
    i,
    j,
    Nc,
    Dc,
    ND_obs,
    V,
    # Coords
    D,
    dD,
    # Error options
    target,
    transformation,
    error_metric,
    censoring,
):
    """Estimate NormalizedGeneralizedGammaPSD model parameters using Grid Search."""
    # Thurai 2018: mu [-3, 1], c [0-6]
    
    # Define parameters values
    mu_values = np.arange(-7, 30, step=0.01)
    c_values = np.arange(0.01, 10, step=0.01)
    
    # Define combinations of parameters for grid search
    mu_grid, c_grid = np.meshgrid(
        mu_values,
        c_values,
        indexing="xy",
    )
    mu_arr = mu_grid.ravel()
    c_arr = c_grid.ravel()

    # Perform grid search
    with suppress_warnings():
        # Compute ND
        ND_preds = NormalizedGeneralizedGammaPSD.formula(
            D=D[None, :], i=i, j=j, Nc=Nc, Dc=Dc, mu=mu_arr[:, None], c=c_arr[:, None]
        )                
        # Compute errors
        errors = compute_cost_function(
            ND_obs=ND_obs/Nc  if target == "H(x)" else ND_obs,
            ND_preds=ND_preds/Nc  if target == "H(x)" else ND_preds,
            D=D,
            dD=dD,
            V=V,
            target=target,
            transformation=transformation,
            error_metric=error_metric,
            censoring=censoring,
        )

    # Replace inf with NaN
    errors[~np.isfinite(errors)] = np.nan

    # If all invalid, return NaN parameters
    if np.all(np.isnan(errors)):
        return np.array([np.nan, np.nan])

    # Otherwise, choose the best index
    best_index = np.nanargmin(errors)
    mu, c = mu_arr[best_index].item(), c_arr[best_index].item()
    return np.array([Nc, Dc, mu, c])
