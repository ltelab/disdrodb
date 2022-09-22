import os
import sys
import logging


logger = logging.getLogger(__name__)


def get_all_readers() -> dict:
    """This function returns the list of reader included into the current release of DISDRODB.

    Returns
    -------
    dict
        The dictionary has the following schema {"data_source":{"campaign_name":"reader file path"}}
    """
    # current file path
    lo_folder_path = os.path.dirname(__file__)

    # readers folder path
    reader_folder_path = os.path.join(lo_folder_path, "readers")

    # list of readers folder
    list_of_reader_folder = [
        f.path for f in os.scandir(reader_folder_path) if f.is_dir()
    ]

    # create dictionary
    dict_reader = {}
    for path_folder in list_of_reader_folder:
        data_source = os.path.basename(path_folder)
        dict_reader[data_source] = {}
        for path_python_file in [
            f.path
            for f in os.scandir(path_folder)
            if f.is_file() and f.path.endswith(".py")
        ]:
            reader_name = (
                os.path.basename(path_python_file)
                .replace("reader_", "")
                .replace(".py", "")
            )
            dict_reader[data_source][reader_name] = path_python_file

    return dict_reader


def get_reader(data_source: str, reader_name: str) -> object:
    """Returns the reader function base on input parameters

    Parameters
    ----------
    data_source : str
        Institution name (when campaign data spans more than 1 country) or country (when all campaigns (or sensor networks) are inside a given country)
    reader_name : str
        Campaign name

    Returns
    -------
    object
        The reader() function

    Raises
    ------
    ValueError
        Error if the data source has not been found within the current readers
    ValueError
        Error if the reader name has not been found within the current readers
    """

    dict_all_readers = get_all_readers()

    if dict_all_readers.get(data_source, None):
        if dict_all_readers.get(data_source).get(reader_name, None):
            current_path = dict_all_readers.get(data_source).get(reader_name)
        else:
            current_path = None
            msg = (
                f"Reader {reader_name} has not been found within the available readers"
            )
            logger.exception(msg)
            raise ValueError(msg)
    else:
        current_path = None
        msg = (
            f"Data source {data_source} has not been found within the available readers"
        )
        logger.exception(msg)
        raise ValueError(msg)

    if current_path:
        full_name = "disdrodb.L0.readers.NETHERLANDS.DELFT.reader"
        module_name, unit_name = full_name.rsplit(".", 1)
        my_function = getattr(__import__(module_name, fromlist=[""]), unit_name)

    return my_function


if __name__ == "__main__":
    get_reader("NETHERLANDS", "DELFT")
