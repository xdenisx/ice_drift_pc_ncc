import os
import re
from pathlib import Path


def _make_source_files_path(drift_infile):
    path = Path(drift_infile).resolve()
    grand_pa = path.parent.parent
    return str(grand_pa / 'S1_names.txt')


def _read_source_files_names(source_files_path):
    with open(source_files_path, 'r') as f:
        return f.read().splitlines()


def _parse_attributes(file_name):
    match = re.search('_(\d{4})(\d{2})(\d{2})T\d{6}_(\d{4})(\d{2})(\d{2})T', file_name)
    if not match:
        raise Exception(f'Cannot extract attributes fron file name {file_name}')
    return {
        'year': match.group(1),
        'month': match.group(2),
        'day': match.group(3)
    }


def make_attributes(data_file):
    source_files = _read_source_files_names(
        _make_source_files_path(
            data_file
        )
    )
    attr1 = _parse_attributes(source_files[0])
    attr2 = _parse_attributes(source_files[1])
    return {
        'source1': source_files[0],
        'source2': source_files[1],
        'year1': attr1['year'],
        'month1': attr1['month'],
        'day1': attr1['day'],
        'year2': attr2['year'],
        'month2': attr2['month'],
        'day2': attr2['day'],
    }