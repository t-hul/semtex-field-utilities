import argparse
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np

from semtex_fieldio.fieldfile import Fieldfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def weighted_subtract(
        ff1: Fieldfile, zero_steps: int, steps_only: bool = False
) -> tuple[np.ndarray, int]:
    steps1 = int(ff1.hdr.step)
    steps1 = read_avg_if_zero(steps1, ff1.fname)
    if zero_steps >= steps1:
        raise ValueError("Cannot subtract: file1 must have more steps than file2")
    result_steps = steps1 - zero_steps
    if steps_only:
        logger.info(f"subtracting {zero_steps} from {steps1} steps: {result_steps}")
        return result_steps
    result = (ff1.data * steps1) / result_steps
    return result, result_steps


def main():
    parser = argparse.ArgumentParser(
        description="Scale avg field as if a field of zero was subtracted with n steps"
    )
    parser.add_argument("file1", help="First input field file")
    parser.add_argument("output", help="Output field file")
    parser.add_argument(
        "--zero_steps",
        type=int,
        help="Number of zero-steps to remove from avg",
    )
    args = parser.parse_args()

    # read and write headers
    ff1 = Fieldfile(args.file1, "r")
    result_steps = weighted_subtract(ff1, args.zero_steps, steps_only=True)

    out_hdr = deepcopy(ff1.hdr)
    out_hdr.step = result_steps
    out_file = Fieldfile(args.output, "w", out_hdr)

    # process one field at a time
    for field in out_hdr.fields:
        ff1.data = ff1.read_fields([field])

        result_data, result_steps = weighted_subtract(ff1, args.zero_steps)

        out_file.write(result_data)


if __name__ == "__main__":
    main()
