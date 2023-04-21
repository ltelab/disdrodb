import os
import re


def check_path(path: str) -> None:
    """Check if a path exists.

    Parameters
    ----------
    path : str
        Path to check.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist. Please check the path.")


def check_url(url: str) -> bool:
    """Check url.

    Parameters
    ----------
    url : str
        URL to check.

    Returns
    -------
    bool
        True if url well formated, False if not well formated.
    """
    regex = r"^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"  # noqa: E501

    if re.match(regex, url):
        return True
    else:
        return False


def check_disdrodb_dir(disdrodb_dir: str):
    """Raise an error if the path does not end with "DISDRODB"."""
    if not disdrodb_dir.endswith("DISDRODB"):
        raise ValueError(f"The path {disdrodb_dir} does not end with DISDRODB. Please check the path.")
