import argparse

import numpy as np

from semtex_fieldio.fieldfile import Fieldfile


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


def weighted_add(
    ff1: Fieldfile, ff2: Fieldfile, scale: float
) -> tuple[np.ndarray, int]:
    steps1 = ff1.hdr.step * scale
    steps2 = ff2.hdr.step
    total_steps = steps1 + steps2
    result = (ff1.data * steps1 + ff2.data * steps2) / total_steps
    return result, total_steps


def weighted_subtract(
    ff1: Fieldfile, ff2: Fieldfile, scale: float
) -> tuple[np.ndarray, int]:
    steps1 = ff1.hdr.step * scale
    steps2 = ff2.hdr.step
    if steps2 >= steps1:
        raise ValueError("Cannot subtract: file1 must have more steps than file2")
    result_steps = steps1 - steps2
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", dest="add_file", help="Second field file to add")
    group.add_argument(
        "--subtract", dest="sub_file", help="Second field file to subtract"
    )
    args = parser.parse_args()

    ff1 = Fieldfile(args.file1, "r")

    if args.add_file:
        ff2 = Fieldfile(args.add_file, "r")
        check_compatibility(ff1, ff2)
        result_data, result_steps = weighted_add(ff1, ff2, args.scale_steps)

    elif args.sub_file:
        ff2 = Fieldfile(args.sub_file, "r")
        check_compatibility(ff1, ff2)
        result_data, result_steps = weighted_subtract(ff1, ff2, args.scale_steps)

    else:
        raise RuntimeError("Either --add or --subtract must be provided")

    ff1.hdr.step = result_steps
    out_file = Fieldfile(args.output, "w", ff1.hdr)
    out_file.write(result_data)


if __name__ == "__main__":
    main()
