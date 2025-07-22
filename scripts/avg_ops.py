import argparse
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np

from semtex_fieldio.fieldfile import Fieldfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_compatibility(ff1: Fieldfile, ff2: Fieldfile) -> None:
    if ff1.fields != ff2.fields:
        raise ValueError("Field lists do not match:", ff1.fields, ff2.fields)
    if not ff1.has_equal_geometry(ff2.geometry):
        raise ValueError("Geometry mismatch between field files")
    if ff1.hdr.dt != ff2.hdr.dt:
        raise ValueError(f"Time step mismatch: {ff1.hdr.dt} != {ff2.hdr.dt}")
    if ff1.hdr.kinvis != ff2.hdr.kinvis:
        raise ValueError(f"kinvis mismatch: {ff1.hdr.kinvis} != {ff2.hdr.kinvis}")
    if ff1.hdr.beta != ff2.hdr.beta:
        raise ValueError(f"beta mismatch: {ff1.hdr.beta} != {ff2.hdr.beta}")


def check_data(ff1: Fieldfile, ff2: Fieldfile) -> None:
    if ff1.data is None or ff2.data is None:
        raise ValueError(f"Field data not loaded correctly")
    if ff1.data.shape != ff2.data.shape:
        raise ValueError("Data shape mismatch: {ff1.data.shape} != {ff2.data.shape}")


def read_avg_if_zero(steps: int, path: Path, scale: float = 1) -> int:
    if steps == 0:
        avg_path = path.with_suffix(".avg")
        logger.info(f"Steps in {path.name} is zero, reading header of {avg_path.name}")
        try:
            ff_avg = Fieldfile(avg_path, "r")
            steps = ff_avg.hdr.step
        except:
            logger.info("Could not read {avg_path.name}")
        # fallback to input if still zero
        return int(input_if_zero(steps, avg_path.name) * scale)
    return steps


def input_if_zero(steps: int, name: str) -> int:
    if steps == 0:
        return int(input(f"Steps of {name} are 0. Enter correct number of steps: "))
    return steps


def weighted_add(
    ff1: Fieldfile, ff2: Fieldfile, scale: float, steps_only: bool = False
) -> tuple[np.ndarray, int]:
    steps1 = int(ff1.hdr.step * scale)
    steps1 = read_avg_if_zero(steps1, ff1.fname, scale)
    steps2 = ff2.hdr.step
    steps2 = read_avg_if_zero(steps2, ff2.fname)
    total_steps = steps1 + steps2
    if steps_only:
        logger.info(f"adding {steps1} and {steps2} steps: {total_steps}")
        return total_steps
    result = (ff1.data * steps1 + ff2.data * steps2) / total_steps
    return result, total_steps


def weighted_subtract(
    ff1: Fieldfile, ff2: Fieldfile, scale: float, steps_only: bool = False
) -> tuple[np.ndarray, int]:
    steps1 = int(ff1.hdr.step * scale)
    steps1 = read_avg_if_zero(steps1, ff1.fname, scale)
    steps2 = ff2.hdr.step
    steps2 = read_avg_if_zero(steps2, ff2.fname)
    if steps2 >= steps1:
        raise ValueError("Cannot subtract: file1 must have more steps than file2")
    result_steps = steps1 - steps2
    if steps_only:
        logger.info(f"subtracting {steps2} from {steps1} steps: {result_steps}")
        return result_steps
    result = (ff1.data * steps1 - ff2.data * steps2) / result_steps
    return result, result_steps


def main():
    parser = argparse.ArgumentParser(
        description="Perform arithmetic on average field files"
    )
    parser.add_argument("file1", help="First input field file")
    parser.add_argument("output", help="Output field file")
    parser.add_argument(
        "--scale_steps",
        type=float,
        default=1.0,
        help="Scale factor for file1's step count",
    )
    parser.add_argument(
        "--omit_c",
        action="store_true",
        help="Omit fields related to 'c'. Use when combining different Pr.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", dest="add_file", help="Second field file to add")
    group.add_argument(
        "--subtract", dest="sub_file", help="Second field file to subtract"
    )
    args = parser.parse_args()

    # read and write headers
    ff1 = Fieldfile(args.file1, "r")
    result_steps = ff1.hdr.step
    if args.add_file:
        ff2 = Fieldfile(args.add_file, "r")
        check_compatibility(ff1, ff2)
        result_steps = weighted_add(ff1, ff2, args.scale_steps, steps_only=True)
    elif args.sub_file:
        ff2 = Fieldfile(args.sub_file, "r")
        check_compatibility(ff1, ff2)
        result_steps = weighted_subtract(ff1, ff2, args.scale_steps, steps_only=True)
    else:
        raise RuntimeError("Either --add or --subtract must be provided")

    out_hdr = deepcopy(ff1.hdr)
    out_hdr.step = result_steps
    if args.omit_c:
        out_hdr.fields = [
            f for f in ff1.hdr.fields if "c" not in f and f not in ["G", "H", "I", "J"]
        ]
        logger.info(
            f"omit_c: Omitting fields: {[f for f in ff1.fields if f not in out_hdr.fields]}"
        )
    out_file = Fieldfile(args.output, "w", out_hdr)

    # process one field at a time
    for field in out_hdr.fields:
        ff1.data = ff1.read_fields([field])
        if args.add_file:
            ff2.data = ff2.read_fields([field])
            check_data(ff1, ff2)
            result_data, result_steps = weighted_add(ff1, ff2, args.scale_steps)

        elif args.sub_file:
            ff2.data = ff2.read_fields([field])
            check_data(ff1, ff2)
            result_data, result_steps = weighted_subtract(ff1, ff2, args.scale_steps)

        out_file.write(result_data)


if __name__ == "__main__":
    main()
