import zipfile


def _unzip_file(file_path: str, dest_path: str) -> None:
    """Unzip a file into a folder
    Parameters
    ----------
    file_path : str
        Path of the file to unzip
    dest_path : str
        Path of the destination folder
    """

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)
