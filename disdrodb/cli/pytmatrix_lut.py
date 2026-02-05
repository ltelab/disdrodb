import os

import click

from disdrodb.scattering.routines import calculate_scatterer


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--frequency", type=float, required=True, help="Radar frequency [GHz]")
@click.option("--num-points", type=int, default=1024, show_default=True)
@click.option("--diameter-min", type=float, default=0, show_default=True, help="Min diameter [mm]")
@click.option("--diameter-max", type=float, default=8.0, show_default=True, help="Max diameter [mm]")
@click.option("--canting-angle-std", type=float, default=7.0, show_default=True, help="Canting angle std [deg]")
@click.option("--axis-ratio-model", type=str, default="Thurai2007", show_default=True)
@click.option("--permittivity-model", type=str, default="Turner2016", show_default=True)
@click.option("--water-temperature", type=float, default=10.0, show_default=True, help="Water temperature [°C]")
@click.option("--elevation-angle", type=float, default=0.0, show_default=True)
@click.argument("output", type=click.Path(dir_okay=False))
def main(
    frequency,
    num_points,
    diameter_min,
    diameter_max,
    canting_angle_std,
    axis_ratio_model,
    permittivity_model,
    water_temperature,
    elevation_angle,
    output,
):
    """
    Compute ONE pyTMatrix scattering LUT in a fresh Python process.

    OUTPUT is the full path to the .pkl LUT file.
    """
    click.echo("Initializing pyTMatrix LUT generation...")

    scatterer = calculate_scatterer(
        frequency=frequency,
        num_points=num_points,
        diameter_min=diameter_min,
        diameter_max=diameter_max,
        canting_angle_std=canting_angle_std,
        axis_ratio_model=axis_ratio_model,
        water_temperature=water_temperature,
        permittivity_model=permittivity_model,
        elevation_angle=elevation_angle,
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    scatterer.psd_integrator.save_scatter_table(output)

    click.echo(f"✔ LUT written to {output}")
