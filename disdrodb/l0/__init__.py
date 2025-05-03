from disdrodb.l0.l0_reader import available_readers
from disdrodb.routines import (
    run_disdrodb_l0,
    run_disdrodb_l0_station,
    run_disdrodb_l0a,
    run_disdrodb_l0a_station,
    run_disdrodb_l0b,
    run_disdrodb_l0b_station,
    run_disdrodb_l0c,
    run_disdrodb_l0c_station,
)

__all__ = [
    "available_readers",
    # Functions invoking the disdrodb_run_* scripts in the terminal
    "run_disdrodb_l0a_station",
    "run_disdrodb_l0b_station",
    "run_disdrodb_l0c_station",
    "run_disdrodb_l0_station",
    "run_disdrodb_l0",
    "run_disdrodb_l0a",
    "run_disdrodb_l0b",
    "run_disdrodb_l0c",
]
