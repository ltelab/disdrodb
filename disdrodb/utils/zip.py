import shutil
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


def _zip_dir(dir_path: str) -> str:
    """Zip a directory into a file located in the same directory.

    Parameters
    ----------
    dir_path : str
        Path of the directory to zip
    """
    output_path = dir_path + ".zip"
    shutil.make_archive(output_path, "zip", dir_path)
    return output_path
