import re

# TODO: Write to YAML

config = {
    "global_settings": {
        "time_integration": ["10S", "30S", "1MIN", "5MIN", "10MIN", "15MIN", "30MIN", "1H", "ROLL5MIN", "ROLL10MIN"],
        # L2M options
        "l2m_options": {
            "fall_velocity_method": "Beard1976",
            "diameter_min": 0,
            "diameter_max": 8,
            "diameter_spacing": 0.05,
        },
        # PSD models fitting options
        "psd_models": {
            "gamma": {
                "probability_method": "cdf",
                "likelihood": "multinomial",
                "truncated_likelihood": True,
                "optimizer": "Nelder-Mead",
                "add_gof_metrics": True,
            },
            "normalized_gamma": {
                "optimizer": "Nelder-Mead",
                "order": 2,
                "add_gof_metrics": True,
            },
            # 'fixed': {
            #     'psd_model': 'NormalizedGammaPSD',
            #     'parameters': {
            #         'mu': [1.5, 2, 2.5, 3],
            #     },
            #     "add_gof_metrics": True
            # },
        },
        # Radar options
        "radar_simulation_enabled": True,
        "radar_simulation_options": {
            "radar_band": ["S", "C", "X", "Ku", "Ka", "W"],
            "canting_angle_std": 7,
            "diameter_max": 8,
            "axis_ratio": "Thurai2007",
        },
    },
    "specific_settings": {
        "10S": {
            "radar_simulation_enabled": False,
        },
        "30S": {
            "radar_simulation_enabled": False,
        },
        "10MIN": {
            "radar_simulation_enabled": False,
        },
        "15MIN": {
            "radar_simulation_enabled": False,
        },
        "30MIN": {
            "radar_simulation_enabled": False,
        },
        "1H": {
            "radar_simulation_enabled": False,
        },
        "ROLL10MIN": {
            "radar_simulation_enabled": False,
        },
    },
}


def get_l2_processing_options():
    """Retrieve L2 processing options."""
    # TODO: Implement validation !
    l2_options_dict = {}
    for tt in config["global_settings"]["time_integration"]:
        l2_options_dict[tt] = config["global_settings"].copy()
        _ = l2_options_dict[tt].pop("time_integration", None)
    for tt, product_options in config["specific_settings"].items():
        l2_options_dict[tt].update(product_options)
    return l2_options_dict


def get_resampling_information(sample_interval_acronym):
    """
    Extract resampling information from the sample interval acronym.

    Parameters
    ----------
    sample_interval_acronym: str
      A string representing the sample interval: e.g., "1H30MIN", "ROLL1H30MIN".

    Returns
    -------
    sample_interval_seconds, rolling: tuple
        Sample_interval in seconds and whether rolling is enabled.
    """
    rolling = sample_interval_acronym.startswith("ROLL")
    if rolling:
        sample_interval_acronym = sample_interval_acronym[4:]  # Remove "ROLL"

    # Regular expression to match duration components
    pattern = r"(\d+)([DHMIN]+)"
    matches = re.findall(pattern, sample_interval_acronym)

    # Conversion factors for each unit
    unit_to_seconds = {
        "D": 86400,  # Seconds in a day
        "H": 3600,  # Seconds in an hour
        "MIN": 60,  # Seconds in a minute
        "S": 1,  # Seconds in a second
    }

    # Parse matches and calculate total seconds
    sample_interval = 0
    for value, unit in matches:
        value = int(value)
        if unit in unit_to_seconds:
            sample_interval += value * unit_to_seconds[unit]
    return sample_interval, rolling
