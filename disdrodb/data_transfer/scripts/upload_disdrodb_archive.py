import click

from disdrodb.data_transfer.upload_data import click_upload_option


@click.command()
@click_upload_option
def upload_disdrodb_archive(
    base_dir=None,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    platform=None,
    files_compression=None,
    force=False,
):
    from disdrodb.data_transfer.upload_data import upload_disdrodb_archives
    from disdrodb.utils.scripts import parse_arg_to_list

    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    upload_disdrodb_archives(
        platform=platform,
        files_compression=files_compression,
        force=force,
        base_dir=base_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
    )
