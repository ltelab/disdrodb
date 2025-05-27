import os

# TODO: Write to YAML
# TODO: radar_simulation_enabled: differentiate between L2E and L2M:

DEFAULT_CONFIG = {
    "global_settings": {
        "time_integration": [
            "1MIN",
            "5MIN",
            "10MIN",
            "ROLL1MIN",
        ],  # ["10S", "30S", "1MIN",  "5MIN", "10MIN", "15MIN", "30MIN", "1H", "ROLL5MIN", "ROLL10MIN"],
        # Radar options
        "radar_simulation_enabled": False,
        "radar_simulation_options": {
            "radar_band": ["S", "C", "X", "Ku", "Ka", "W"],
            "canting_angle_std": 7,
            "diameter_max": 10,
            "axis_ratio": "Thurai2007",
        },
        # L2E options
        # "l2e_options": {}
        # L2M options
        "l2m_options": {
            "fall_velocity_method": "Beard1976",
            "diameter_min": 0,
            "diameter_max": 10,
            "diameter_spacing": 0.05,
            "gof_metrics": True,
            "min_nbins": 4,
            "remove_timesteps_with_few_bins": False,
            "mask_timesteps_with_few_bins": False,
            "models": {
                # PSD models fitting options
                "GAMMA_ML": {
                    "psd_model": "GammaPSD",
                    "optimization": "ML",
                    "optimization_kwargs": {
                        "init_method": "M346",
                        "probability_method": "cdf",
                        "likelihood": "multinomial",
                        "truncated_likelihood": True,
                        "optimizer": "Nelder-Mead",
                    },
                },
                "NGAMMA_GS_LOG_ND_MAE": {
                    "psd_model": "NormalizedGammaPSD",
                    "optimization": "GS",
                    "optimization_kwargs": {
                        "target": "ND",
                        "transformation": "log",
                        "error_order": 1,  # MAE
                    },
                },
                # "NGAMMA_GS_ND_MAE": {
                #     "psd_model": "NormalizedGammaPSD",
                #     "optimization": "GS",
                #     "optimization_kwargs": {
                #         "target": "ND",
                #         "transformation": "identity",
                #         "error_order": 1,  # MAE
                #     },
                # },
                # "NGAMMA_GS_Z": {
                #     "psd_model": "NormalizedGammaPSD",
                #     "optimization": "GS",
                #     "optimization_kwargs": {
                #         "target": "Z",
                #         "transformation": "identity",  # unused
                #         "error_order": 1,  # unused
                #     },
                # },
            },
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

TEST_CONFIG = {
    "global_settings": {
        "time_integration": [
            "1MIN",
            "10MIN",
            "ROLL1MIN",
            "ROLL10MIN",
        ],  # ["10S", "30S", "1MIN",  "5MIN", "10MIN", "15MIN", "30MIN", "1H", "ROLL5MIN", "ROLL10MIN"],
        # Radar options
        "radar_simulation_enabled": False,
        "radar_simulation_options": {
            "radar_band": ["S", "C", "X", "Ku", "Ka", "W"],
            "canting_angle_std": 7,
            "diameter_max": 10,
            "axis_ratio": "Thurai2007",
        },
        # L2E options
        # "l2e_options": {}
        # L2M options
        "l2m_options": {
            "fall_velocity_method": "Beard1976",
            "diameter_min": 0,
            "diameter_max": 10,
            "diameter_spacing": 0.05,
            "gof_metrics": True,
            "min_nbins": 4,
            "remove_timesteps_with_few_bins": False,
            "mask_timesteps_with_few_bins": False,
            "models": {
                # PSD models fitting options
                "GAMMA_ML": {
                    "psd_model": "GammaPSD",
                    "optimization": "ML",
                    "optimization_kwargs": {
                        "init_method": "M346",
                        "probability_method": "cdf",
                        "likelihood": "multinomial",
                        "truncated_likelihood": True,
                        "optimizer": "Nelder-Mead",
                    },
                },
                "NGAMMA_GS_LOG_ND_MAE": {
                    "psd_model": "NormalizedGammaPSD",
                    "optimization": "GS",
                    "optimization_kwargs": {
                        "target": "ND",
                        "transformation": "log",
                        "error_order": 1,  # MAE
                    },
                },
                # "NGAMMA_GS_ND_MAE": {
                #     "psd_model": "NormalizedGammaPSD",
                #     "optimization": "GS",
                #     "optimization_kwargs": {
                #         "target": "ND",
                #         "transformation": "identity",
                #         "error_order": 1,  # MAE
                #     },
                # },
                # "NGAMMA_GS_Z": {
                #     "psd_model": "NormalizedGammaPSD",
                #     "optimization": "GS",
                #     "optimization_kwargs": {
                #         "target": "Z",
                #         "transformation": "identity",  # unused
                #         "error_order": 1,  # unused
                #     },
                # },
            },
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
    # Define config to use
    config = TEST_CONFIG if os.environ.get("PYTEST_CURRENT_TEST") else DEFAULT_CONFIG
    # Define global L2 options
    for tt in config["global_settings"]["time_integration"]:
        l2_options_dict[tt] = config["global_settings"].copy()
        _ = l2_options_dict[tt].pop("time_integration", None)
    # Add specific settings
    for tt, product_options in config["specific_settings"].items():
        if tt in l2_options_dict:
            l2_options_dict[tt].update(product_options)
    return l2_options_dict
