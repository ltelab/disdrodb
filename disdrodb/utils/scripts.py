#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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


def _execute_cmd(cmd, raise_error=False):
    """Execute command in the terminal, streaming output in python console."""
    from subprocess import Popen, PIPE, CalledProcessError

    with Popen(cmd, shell=True, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end="")

    # Raise error if command didn't run successfully
    if p.returncode != 0 and raise_error:
        raise CalledProcessError(p.returncode, p.args)


def parse_arg_to_list(args):
    """Utility to pass list to command line scripts.

    If args = '' --> None
    If args = 'variable' --> [variable]
    If args = 'variable1 variable2' --> [variable1, variable2]
    """
    # If '', set to None
    args = None if args == "" else args
    # - If multiple arguments, split by space
    if isinstance(args, str):
        # - Split by space
        list_args = args.split(" ")
        # - Remove '' (deal with multi space)
        args = [args for args in list_args if len(args) > 0]
    return args
