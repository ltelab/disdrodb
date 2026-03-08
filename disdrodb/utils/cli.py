# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""DISDRODB command-line-interface scripts utilities."""

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import click


def _env_scripts_dir() -> Path:
    """Return the directory where console scripts for the *current* Python environment live.

    Works for conda envs and venvs on Linux/macOS/Windows.

    Returns
    -------
    Path
        The directory path where console scripts are installed.

    Raises
    ------
    FileNotFoundError
        If the scripts directory cannot be determined on Windows.
    """
    exe = Path(sys.executable).resolve()

    if os.name == "nt":
        parent = exe.parent

        # venv/virtualenv layout: <venv>\Scripts\python.exe  -> <venv>\Scripts
        if parent.name.lower() == "scripts" and parent.exists():
            return parent

        # conda-style layout: <env>\python.exe -> <env>\Scripts
        cand = parent / "Scripts"
        if cand.exists():
            return cand

        # less common: <env>\bin\python.exe (msys/cygwin style)
        if parent.name.lower() == "bin" and parent.exists():
            return parent

        # Raise error otherwise
        raise FileNotFoundError(
            f"Could not determine scripts directory for Windows environment. Checked: {parent} and {cand}",
        )

    # POSIX: typically .../env/bin/python -> .../env/bin
    return exe.parent


def get_command_argv(cmd):
    """Parse command into argument list.

    Parameters
    ----------
    cmd : str or Sequence[str]
        Command to parse. Can be a string (will be split using shell-like syntax)
        or a sequence of strings.

    Returns
    -------
    list
        List of command arguments.

    Raises
    ------
    ValueError
        If the command is empty.
    """
    shell = isinstance(cmd, str)
    argv = shlex.split(cmd) if shell else cmd
    if not argv:
        raise ValueError("cmd must be non-empty")
    return argv


def retrieve_environment():
    """Retrieve environment variables with scripts directory added to PATH.

    Creates a copy of the current environment and ensures that the Python
    environment's scripts directory is on the PATH.

    Returns
    -------
    dict
        Dictionary of environment variables with updated PATH.
    """
    # Retrieve scripts directory
    env = os.environ.copy()
    scripts_dir = str(_env_scripts_dir())

    # Add scripts_dir to PATH only if not already present
    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    if scripts_dir not in path_parts:
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    return env


def retrieve_executable(command, env):
    """Resolve the full path to an executable command.

    Parameters
    ----------
    command : str
        The command name to resolve.
    env : dict
        Environment variables dictionary containing PATH.

    Returns
    -------
    str
        Full path to the executable.

    Raises
    ------
    FileNotFoundError
        If the command cannot be found on PATH.
    """
    # Resolve the executable explicitly (more deterministic)
    exe = shutil.which(command, path=env["PATH"])
    if exe is None:
        raise FileNotFoundError(
            f"Command '{command}' not found on PATH for kernel environment."
            f"Tried searching in the following directories: {env['PATH'].split(':')}",
        )
    return exe


def subprocess_run(
    cmd: str | Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    cwd: str | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a command ensuring the current kernel's env 'bin/Scripts' is on PATH.

    This wrapper ensures subprocess can find and run console scripts from the
    current Jupyter kernel's conda/venv by adding that environment's bin/
    (or Scripts/ on Windows) to PATH when the notebook starts with a
    system-only PATH.

    Parameters
    ----------
    cmd : str or Sequence[str]
        Command to execute. Can be a string or list of arguments.
        Example: ["disdrodb_run_l0", "--help"]
    check : bool, optional
        If True, raise CalledProcessError if the process returns a non-zero exit code.
        Default is True.
    capture_output : bool, optional
        If True, capture stdout and stderr. Default is False.
    text : bool, optional
        If True, decode stdout/stderr as text. Default is True.
    cwd : str or None, optional
        Working directory for the command. Default is None.
    **kwargs
        Additional keyword arguments passed to subprocess.run.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process object containing return code, stdout, and stderr.

    Raises
    ------
    FileNotFoundError
        If the command executable cannot be found.
    subprocess.CalledProcessError
        If check=True and the process returns a non-zero exit code.
    """
    # Retrieve command arguments as list (shell=False)
    argv = get_command_argv(cmd)
    _ = kwargs.pop("shell", None)

    # Retrieve environment
    env = retrieve_environment()

    # Retrieve executable path
    exe = retrieve_executable(command=argv[0], env=env)

    # Define list of commands to run
    cmd = [exe, *argv[1:]]

    # Run command
    return subprocess.run(
        cmd,
        shell=False,
        env=env,
        check=check,
        capture_output=capture_output,
        text=text,
        cwd=cwd,
        **kwargs,
    )


def execute_cmd(cmd: str | Sequence[str], raise_error: bool = False, cwd: str | None = None, **kwargs):
    """Execute a command, streaming output live in the Python console.

    Ensures the current kernel environment's bin/ (or Scripts/ on Windows)
    is added to PATH so console scripts installed in the active venv/conda
    environment can be found.

    Parameters
    ----------
    cmd : str or Sequence[str]
        Command to execute. Can be a string or list of arguments.
    raise_error : bool, optional
        If True, raise CalledProcessError on non-zero exit code. Default is False.
    cwd : str or None, optional
        Working directory for the command. Default is None.
    **kwargs
        Additional keyword arguments passed to subprocess.Popen.

    Returns
    -------
    int
        The return code of the executed command.

    Raises
    ------
    FileNotFoundError
        If the command executable cannot be found.
    subprocess.CalledProcessError
        If raise_error=True and the command returns a non-zero exit code.
    """
    from subprocess import PIPE, STDOUT, CalledProcessError, Popen

    # Retrieve command arguments as list (shell=False)
    argv = get_command_argv(cmd)
    _ = kwargs.pop("shell", None)

    # Retrieve environment
    env = retrieve_environment()

    # Retrieve executable path
    exe = retrieve_executable(command=argv[0], env=env)

    # Define list of commands to run
    cmd = [exe, *argv[1:]]

    # Execute command
    with Popen(
        cmd,
        shell=False,
        stdout=PIPE,
        stderr=STDOUT,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        env=env,
        **kwargs,
    ) as p:
        if p.stdout is not None:
            for line in p.stdout:
                print(line, end="")
        returncode = p.wait()

    # Raise error if command didn't run successfully
    if p.returncode != 0 and raise_error:
        raise CalledProcessError(returncode, cmd)
    return returncode


def parse_empty_string_and_none(args):
    """Parse command line argument handling empty strings and 'None'.

    Parameters
    ----------
    args : str or any
        Argument to parse.

    Returns
    -------
    any or None
        Returns None if args is '' or 'None', otherwise returns args unchanged.

    Examples
    --------
    >>> parse_empty_string_and_none('')
    None
    >>> parse_empty_string_and_none('None')
    None
    >>> parse_empty_string_and_none('value')
    'value'
    """
    # If '', set to 'None'
    args = None if args == "" else args
    # - If multiple arguments, split by space
    if isinstance(args, str) and args == "None":
        args = None
    return args


def parse_arg_to_list(args):
    """Parse space-separated command line argument into a list.

    Parameters
    ----------
    args : str or any
        Space-separated string or other argument type.

    Returns
    -------
    list or None
        List of parsed arguments, or None if args is empty or 'None'.

    Examples
    --------
    >>> parse_arg_to_list('')
    None
    >>> parse_arg_to_list('None')
    None
    >>> parse_arg_to_list('variable')
    ['variable']
    >>> parse_arg_to_list('variable1 variable2')
    ['variable1', 'variable2']
    """
    # If '' or 'None' --> Set to None
    args = parse_empty_string_and_none(args)
    # - If multiple arguments, split by space
    if isinstance(args, str):
        # - Split by space
        list_args = args.split(" ")
        # - Remove '' (deal with multi space)
        args = [args for args in list_args if len(args) > 0]
    return args


def parse_archive_dir(archive_dir: str):
    """Parse archive directory from command line argument.

    Parameters
    ----------
    archive_dir : str
        Archive directory path as string.

    Returns
    -------
    str or None
        Archive directory path, or None if archive_dir is empty or 'None'.

    Examples
    --------
    >>> parse_archive_dir('')
    None
    >>> parse_archive_dir('None')
    None
    >>> parse_archive_dir('/path/to/archive')
    '/path/to/archive'
    """
    # If '', set to 'None'
    return parse_empty_string_and_none(archive_dir)


def click_station_arguments(function: object):
    """Add Click command line arguments for DISDRODB station processing.

    Adds three required arguments: data_source, campaign_name, and station_name.

    Parameters
    ----------
    function : callable
        Function to decorate with Click arguments.

    Returns
    -------
    callable
        Decorated function with Click arguments added.
    """
    function = click.argument("station_name", metavar="<STATION_NAME>")(function)
    function = click.argument("campaign_name", metavar="<CAMPAIGN_NAME>")(function)
    function = click.argument("data_source", metavar="<DATA_SOURCE>")(function)
    return function


def click_data_archive_dir_option(function: object):
    """Add Click command line option for DISDRODB data archive directory.

    Adds the --data_archive_dir option for specifying the DISDRODB data
    archive directory path.

    Parameters
    ----------
    function : callable
        Function to decorate with Click option.

    Returns
    -------
    callable
        Decorated function with --data_archive_dir option added.
    """
    function = click.option(
        "--data_archive_dir",
        type=str,
        show_default=True,
        default=None,
        help="DISDRODB Data Archive Directory. Format: <...>/DISDRODB",
    )(function)
    return function


def click_metadata_archive_dir_option(function: object):
    """Add Click command line option for DISDRODB metadata archive directory.

    Adds the --metadata_archive_dir option for specifying the DISDRODB metadata
    archive directory path.

    Parameters
    ----------
    function : callable
        Function to decorate with Click option.

    Returns
    -------
    callable
        Decorated function with --metadata_archive_dir option added.
    """
    function = click.option(
        "--metadata_archive_dir",
        type=str,
        show_default=True,
        default=None,
        help="DISDRODB Metadata Archive Directory. Format: <...>/DISDRODB-METADATA/DISDRODB",
    )(function)
    return function


def click_stations_options(function: object):
    """Add Click command line options for filtering DISDRODB stations.

    Adds options for --data_sources, --campaign_names, and --station_names
    to filter which stations to process.

    Parameters
    ----------
    function : callable
        Function to decorate with Click options.

    Returns
    -------
    callable
        Decorated function with station filtering options added.
    """
    function = click.option(
        "--data_sources",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB data sources to process",
    )(function)
    function = click.option(
        "--campaign_names",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB campaign names to process",
    )(function)
    function = click.option(
        "--station_names",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB station names to process",
    )(function)
    return function


def click_processing_options(function: object):
    """Add Click command line options for DISDRODB processing configuration.

    Adds options for --parallel, --debugging_mode, --verbose, and --force
    to control processing behavior.

    Parameters
    ----------
    function : callable
        Function to decorate with Click options.

    Returns
    -------
    callable
        Decorated function with processing configuration options added.
    """
    function = click.option(
        "-p",
        "--parallel",
        type=bool,
        show_default=True,
        default=False,
        help="Process files in parallel",
    )(function)
    function = click.option(
        "-d",
        "--debugging_mode",
        type=bool,
        show_default=True,
        default=False,
        help="Switch to debugging mode",
    )(function)
    function = click.option("-v", "--verbose", type=bool, show_default=True, default=True, help="Verbose")(function)
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=False,
        help="Force overwriting",
    )(function)
    return function


def click_remove_l0a_option(function: object):
    """Add Click command line option for removing L0A files.

    Adds the --remove_l0a boolean option to control whether L0A files
    should be removed after L0B processing.

    Parameters
    ----------
    function : callable
        Function to decorate with Click option.

    Returns
    -------
    callable
        Decorated function with --remove_l0a option added.
    """
    function = click.option(
        "--remove_l0a",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0A files once the L0B processing is terminated.",
    )(function)
    return function


def click_remove_l0b_option(function: object):
    """Add Click command line option for removing L0B files.

    Adds the --remove_l0b boolean option to control whether L0B files
    should be removed after L0C processing.

    Parameters
    ----------
    function : callable
        Function to decorate with Click option.

    Returns
    -------
    callable
        Decorated function with --remove_l0b option added.
    """
    function = click.option(
        "--remove_l0b",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0B files once the L0C processing is terminated.",
    )(function)
    return function


def click_l0_archive_options(function: object):
    """Add Click command line options for L0 processing and archiving control.

    Adds options to control which L0 processing levels to run (L0A, L0B, L0C)
    and whether to remove intermediate files.

    Parameters
    ----------
    function : callable
        Function to decorate with Click options.

    Returns
    -------
    callable
        Decorated function with L0 archive options added.
    """
    function = click.option(
        "--remove_l0b",
        type=bool,
        show_default=True,
        default=False,
        help="If True, remove L0B files after L0C.",
    )(function)
    function = click.option(
        "--remove_l0a",
        type=bool,
        show_default=True,
        default=False,
        help="If True, remove L0A files after L0B.",
    )(function)
    function = click.option(
        "-l0c",
        "--l0c_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Run L0C processing",
    )(function)
    function = click.option(
        "-l0b",
        "--l0b_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Run L0B processing",
    )(function)
    function = click.option(
        "-l0a",
        "--l0a_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Run L0A processing",
    )(function)
    return function
