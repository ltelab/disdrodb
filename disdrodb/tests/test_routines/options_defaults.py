# Base configuration components
ARCHIVE_OPTIONS_TIME_BLOCK = {
    "strategy": "time_block",
    "strategy_options": {"freq": "day"},
    "folder_partitioning": "year",
}

ARCHIVE_OPTIONS_EVENT = {
    "strategy": "event",
    "strategy_options": {
        "variable": "n_particles",
        "detection_threshold": 10,
        "neighbor_min_size": 5,
        "neighbor_time_interval": "1MIN",
        "event_max_time_gap": "5MIN",
        "event_min_duration": "10MIN",
        "event_min_size": 20,
    },
    "folder_partitioning": "year/month",
}

RADAR_OPTIONS = {
    "frequency": ["S", "C", "X", "Ku", "K", "Ka", "W"],
    "num_points": 1024,
    "diameter_max": 10,
    "canting_angle_std": 7,
    "axis_ratio_model": "Thurai2007",
    "permittivity_model": "Turner2016",
    "water_temperature": 10,
    "elevation_angle": 0,
}

L2E_PRODUCT_OPTIONS = {
    "compute_spectra": False,
    "compute_percentage_contribution": False,
    "minimum_ndrops": 1,
    "minimum_nbins": 1,
    "minimum_rain_rate": 0.01,
    "fall_velocity_model": "Beard1976",
    "minimum_diameter": 0,
    "maximum_diameter": 10,
    "minimum_velocity": 0,
    "maximum_velocity": 12,
    "keep_mixed_precipitation": True,
    "above_velocity_fraction": None,
    "above_velocity_tolerance": 3,
    "below_velocity_fraction": None,
    "below_velocity_tolerance": 3,
    "maintain_drops_smaller_than": 1,
    "maintain_drops_slower_than": 2.5,
    "maintain_smallest_drops": True,
    "remove_splashing_drops": True,
}

L2M_PRODUCT_OPTIONS = {
    "fall_velocity_model": "Beard1976",
    "diameter_min": 0,
    "diameter_max": 10,
    "diameter_spacing": 0.05,
    "gof_metrics": True,
    "minimum_ndrops": 1,
    "minimum_nbins": 5,
    "minimum_rain_rate": 0.01,
}

# Complete product configurations
L0C_GLOBAL_YAML = {
    "archive_options": ARCHIVE_OPTIONS_TIME_BLOCK,
}

L1_GLOBAL_YAML = {
    "temporal_resolutions": ["1MIN", "5MIN", "10MIN", "ROLL1MIN"],
    "archive_options": ARCHIVE_OPTIONS_TIME_BLOCK,
}

L2E_GLOBAL_YAML = {
    "temporal_resolutions": ["1MIN", "5MIN", "10MIN", "ROLL1MIN"],
    "archive_options": {
        "strategy": "time_block",
        "strategy_options": {"freq": "month"},
        "folder_partitioning": "",
    },
    "product_options": L2E_PRODUCT_OPTIONS,
    "radar_enabled": False,
    "radar_options": RADAR_OPTIONS,
}

L2M_GLOBAL_YAML = {
    "temporal_resolutions": ["1MIN", "5MIN", "10MIN"],
    "models": [
        "GAMMA_ML",
        "GAMMA_GS",
        "LOGNORMAL_ML",
    ],
    "archive_options": {
        "strategy": "time_block",
        "strategy_options": {"freq": "month"},
        "folder_partitioning": "",
    },
    "product_options": L2M_PRODUCT_OPTIONS,
    "radar_enabled": False,
    "radar_options": RADAR_OPTIONS,
}

# L2M Model configurations
GAMMA_ML_CONFIG = {
    "psd_model": "GammaPSD",
    "optimization": "ML",
    "optimization_kwargs": {
        "init_method": None,
        "probability_method": "cdf",
        "likelihood": "multinomial",
        "truncated_likelihood": True,
        "optimizer": "Nelder-Mead",
    },
}

GAMMA_GS_CONFIG = {
    "psd_model": "GammaPSD",
    "optimization": "GS",
    "optimization_kwargs": {
        "target": "N(D)",
        "transformation": "identity",
        "error_order": 1,
        "censoring": "none",
    },
}

LOGNORMAL_ML_CONFIG = {
    "psd_model": "LognormalPSD",
    "optimization": "ML",
    "optimization_kwargs": {
        "init_method": None,
        "probability_method": "cdf",
        "likelihood": "multinomial",
        "truncated_likelihood": True,
        "optimizer": "Nelder-Mead",
    },
}
