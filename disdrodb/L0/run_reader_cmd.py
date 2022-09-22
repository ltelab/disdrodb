import imp
import os
import sys
import click

# from disdrodb.L0 import click_L0_readers_options
from disdrodb.L0.retrive_reader import get_reader

root_path = os.getcwd()
sys.path.insert(0, root_path)

# -------------------------------------------------------------------------.
# CLIck Command Line Interface decorator
@click.command()  # options_metavar='<options>'
@click.argument("data_source", metavar="<data_source>")
@click.argument("reader_name", metavar="<reader_name>")
@click.argument("raw_dir", type=click.Path(exists=True), metavar="<raw_dir>")
@click.argument(
    "processed_dir", type=click.Path(exists=False), metavar="<processed_dir>"
)
@click.argument(
    "l0a_processing", type=click.BOOL, default=True, metavar="<l0a_processing>"
)
@click.argument(
    "l0b_processing", type=click.BOOL, default=True, metavar="<l0b_processing>"
)
@click.argument("keep_l0a", type=click.BOOL, default=True, metavar="<keep_l0a>")
@click.argument("force", type=click.BOOL, default=True, metavar="<force>")
@click.argument("verbose", type=click.BOOL, default=False, metavar="<verbose>")
@click.argument(
    "debugging_mode", type=click.BOOL, default=False, metavar="<debugging_mode>"
)
@click.argument("lazy", type=click.BOOL, default=False, metavar="<lazy>")
@click.argument(
    "single_netcdf", type=click.BOOL, default=True, metavar="<single_netcdf>"
)
# @click.argument('data_source', type=click.Path(exists=True), metavar='<data_source>')


def main(
    data_source,
    reader_name,
    raw_dir,
    processed_dir,
    l0a_processing=True,
    l0b_processing=True,
    keep_l0a=False,
    force=False,
    verbose=False,
    debugging_mode=False,
    lazy=True,
    single_netcdf=True,
):

    reader = get_reader(data_source, reader_name)

    reader(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        l0a_processing=l0a_processing,
        l0b_processing=l0b_processing,
        keep_l0a=keep_l0a,
        force=force,
        verbose=verbose,
        debugging_mode=debugging_mode,
        lazy=lazy,
        single_netcdf=single_netcdf,
    )


if __name__ == "__main__":
    main()
