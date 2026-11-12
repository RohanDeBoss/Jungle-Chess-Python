"""Shared on-disk format helpers for Jungle Chess tablebases."""

import os

import numpy as np


def infer_flat_table_dtype(filename, position_count):
    """Infer the signed DTM dtype from the exact bytes-per-position count."""
    count = int(position_count)
    if count <= 0:
        raise ValueError(f"Invalid table position count: {count}.")

    actual_size = os.path.getsize(filename)
    if actual_size == count:
        return np.int8
    if actual_size == count * 2:
        return np.int16
    raise ValueError(
        f"Unexpected table size for {filename}: expected {count:,} bytes "
        f"(int8) or {count * 2:,} bytes (int16), got {actual_size:,}."
    )


def infer_table_dtype(filename, shape):
    """Infer table dtype for a table whose multidimensional shape is known."""
    position_count = int(np.prod(shape, dtype=np.int64))
    return infer_flat_table_dtype(filename, position_count)
