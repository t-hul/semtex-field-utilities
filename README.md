# semtex-field-utilities

## Description
This project contains python utilities to access and manipulate Semtex field files.
It is based on the scripts provided by Hugh Blackburn and Thomas Albrecht in Semtex.
The original scripts are updated to a mor modern approach.

## Installation using pip install
To install the library plot_utils in edit mode run
```
pip install -e .
```
in the projects root directory.

## List of changes
- split fieldfile classes into separate files: geometry.py, header.py, fieldfile.py
- use dataclasses and type annotations
- use fieldstring naming convention
- added pytest functionality

## Testing
To run the tests execute `pytest` in the projects root directory.
Debugging output can be activated in `pytest.ini`.
Additional data for testing can be found in `tests/data/` 

