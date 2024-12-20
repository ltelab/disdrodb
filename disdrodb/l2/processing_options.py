# TODO: Write to YAML
# TODO: radar_simulation_enabled: differentiate between L2E and L2M:

config = {
    "global_settings": {
        "time_integration": [
            "1MIN",
            "10MIN",
            "ROLL1MIN",
            "ROLL10MIN",
        ],  # ["10S", "30S", "1MIN",  "5MIN", "10MIN", "15MIN", "30MIN", "1H", "ROLL5MIN", "ROLL10MIN"],
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
    # Add specific settings
    for tt, product_options in config["specific_settings"].items():
        if tt in l2_options_dict:
            l2_options_dict[tt].update(product_options)
    return l2_options_dict
