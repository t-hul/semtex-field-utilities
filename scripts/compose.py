#!/usr/bin python
#
# Interactive field file composition utility to combine two field
# files, with field selection.  Alternatively, given a single field
# file, choose (and perhaps rename) selected fields.  Write binary
# file (asked for name).
#
# 1. compose.py file.fld
# 2. compose.py file1.fld file2.fld
#
# Input files are assumed to be in a machine-compatible binary format.
# In the second case, the field files must conform (nr, ns, nel, nz).
#
# This utility cannot reorder output fields, and unless you rename/choose
# output field names appropriately, it possible to repeat field names.
#
# $Id: compose.py,v 9.3 2020/01/23 03:15:43 hmb Exp $
# ----------------------------------------------------------------------------

import argparse
import logging

import numpy as np

from semtex_fieldio.fieldfile import Fieldfile

logger = logging.getLogger(__name__)


def compose_single(file1: Fieldfile):
    print(
        "Input has these fields. Supply a string below to indicate/rename ones you want. Separate fields by comma ','. Indicate unwanted fields by whitespace.:"
    )
    print(",".join(file1.fields))
    required_fields = input()
    wanted = required_fields.split(",")  # should result in list of field names
    if len(wanted) < len(file1.fields):
        print(
            f"{required_fields}: must have same or higher number of fields :{file1.fields}"
        )
        exit(1)

    outfile_name = input("type in an output file name: ")
    newhdr = file1.hdr
    newhdr.fields = [f.strip() for f in wanted if not f.isspace()]
    ofile = Fieldfile(outfile_name, "w", newhdr)

    print(newhdr.fields)
    ofile.data = np.zeros((len(ofile.fields), ofile.npoints))

    j = 0
    # copy wanted fields to new file
    logger.debug(f"shape of file1.data: {file1.data.shape}")
    logger.debug(f"shape of ofile.data: {ofile.data.shape}")
    for i, field in enumerate(file1.fields):
        logger.debug(f"field[{i}]: {field}")
        if not wanted[i].isspace():
            logger.debug(f"copy to ofile.field[{j}]: {ofile.fields[j]}")
            ofile.data[j] = file1.data[i]  # copy data based on index of field, not name
            j += 1  # index of ofile.fields
    # fill additional fields with constant values
    if j < len(ofile.fields):
        for idx in range(j, len(ofile.fields)):
            value = input(f"type const value for field {ofile.fields[idx]}: ")
            ofile.data[idx, :] = value

    ofile.write(ofile.data)


def compose_dual(file1: Fieldfile, file2: Fieldfile):
    print(
        "Input1 has these fields, supply a string below to indicate/rename ones you want:"
    )
    print(file1.hdr.fields)
    required_fields1 = input()
    if len(required_fields1) < len(file1.hdr.fields):
        print(required_fields1, ": cannot be shorter than :", file1.hdr.fields)
        exit(1)
    wanted1 = "".join(re.findall(r"\S+", required_fields1))

    print("Input2 has these fields, supply a string below to")
    print("indicate/rename ones you want:")
    print(file2.hdr.fields)
    required_fields2 = input()
    if len(required_fields2) < len(file2.hdr.fields):
        print(required_fields2, ": cannot be shorter than :", file2.hdr.fields)
        exit(1)
    wanted2 = "".join(re.findall(r"\S+", required_fields2))

    outfile = input("type in an output file name: ")
    newhdr = file1.hdr
    newhdr.fields = wanted1 + wanted2
    print("new file will have fields: ", newhdr.fields)
    ofile = Fieldfile(outfile, "w", newhdr)

    ofile.data = np.zeros((len(ofile.hdr.fields), ofile.ntot))

    j = 0
    for i, field in enumerate(file1.fields):
        buf = file1.f.read(file1.ntot * 8)
        if (required_fields1[i].isspace()) == False:
            ofile.data[j] = np.fromstring(buf, np.float64, -1)
            j += 1
    for i, field in enumerate(file2.fields):
        buf = file2.f.read(file2.ntot * 8)
        if (required_fields2[i].isspace()) == False:
            ofile.data[j] = np.fromstring(buf, np.float64, -1)
            j += 1

    ofile.write(ofile.data)
    file1.close()
    file2.close()
    ofile.close()


def main():
    parser = argparse.ArgumentParser(description="Compose Semtex field files")
    parser.add_argument("file1_name", help="First input field file")
    parser.add_argument(
        "file2_name", nargs="?", help="Second input field file (optional)"
    )
    args = parser.parse_args()

    mode = 1 if args.file2_name is None else 2

    file1 = Fieldfile(args.file1_name, "r")

    if mode == 2:
        file2 = Fieldfile(args.file2_name, "r")
        if not file1.has_equal_geometry(file2.geometry):
            print("The two input files do not conform")
            exit(1)

    if mode == 1:
        compose_single(file1)
    else:
        compose_dual(file1, file2)


if __name__ == "__main__":
    main()
