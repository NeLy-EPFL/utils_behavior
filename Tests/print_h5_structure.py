#!/usr/bin/env python3
"""
Print the structure of an HDF5 (.h5) file: groups, datasets, shapes, dtypes and attributes.

Usage:
    python Tests/print_h5_structure.py /path/to/file.h5

This is a small utility based on the project's existing H5 helpers.
"""
import sys
import h5py
from pathlib import Path
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None


def print_attrs(obj, indent=0):
    for k, v in obj.attrs.items():
        print(" " * indent + f"@{k}: {v}")


def format_array_preview(arr, max_rows=10):
    # Convert to numpy for easy slicing; keep memory usage low by slicing first
    try:
        if isinstance(arr, h5py.Dataset):
            # determine slice for first axis
            if arr.size == 0:
                return None
            # Build an indexer that selects up to max_rows along the last axis if vector-like,
            # otherwise along the first axis.
            if arr.ndim == 0:
                data = arr[()]
                # normalize scalar to 0-d numpy array
                data = np.array(data)
            elif arr.ndim == 1:
                end = min(arr.shape[0], max_rows)
                data = arr[:end]
            else:
                # For multi-d arrays, create a small 2D preview: take up to max_rows along
                # the last axis (assumed time/frames) and collapse remaining axes into columns.
                # Example: shape (1,2,1,161465) -> we want rows ~ along last axis -> (n_rows, 1*2*1)
                last_axis = arr.shape[-1]
                n_rows = min(last_axis, max_rows)
                # build slices: take first indices for all axes except last, and a slice for last
                # indexer like [0, 0, 0, :n_rows] for shape (1,2,1,161465)
                indexer = [0] * (arr.ndim - 1) + [slice(0, n_rows)]
                try:
                    small = arr[tuple(indexer)]
                    # small has shape (dim0, dim1, ..., n_rows) -> move last axis to first and flatten others
                    small = np.asarray(small)
                    # move last axis to front
                    small = np.moveaxis(small, -1, 0)
                    # flatten remaining axes so each row is a vector of features
                    data = small.reshape((small.shape[0], -1))
                except Exception:
                    # fallback to a flattened preview
                    flat_end = min(arr.size, max_rows)
                    data = np.array(arr).ravel()[:flat_end]
        else:
            data = np.asarray(arr)
    except Exception:
        return None

    # Handle bytes/string/object dtypes
    # data may be a python scalar or numpy array; coerce to numpy array for dtype checks
    try:
        npdata = np.asarray(data)
    except Exception:
        npdata = None

    if npdata is not None and (npdata.dtype.type is np.bytes_ or npdata.dtype == object):
        try:
            data = npdata.astype(str)
        except Exception:
            pass

    # If pandas is available, use it to pretty-print
    if pd is not None:
        try:
            if data.ndim == 0:
                return pd.DataFrame([data.tolist()])
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame({"value": data.tolist()})

    # Fallback: return numpy array or list
    return data


def print_h5_item(name, obj, indent=0, preview_rows=10):
    pad = " " * indent
    if isinstance(obj, h5py.Dataset):
        try:
            shape = obj.shape
            dtype = obj.dtype
        except Exception:
            shape = "?"
            dtype = "?"
        print(f"{pad}- Dataset: {name}  shape={shape} dtype={dtype}")
        if obj.attrs:
            print_attrs(obj, indent + 4)

        # skip empty datasets
        if getattr(obj, 'size', 0) == 0:
            print(pad + "  (empty - skipped preview)")
            return

        preview = format_array_preview(obj, max_rows=preview_rows)
        if preview is None:
            print(pad + "  (could not create preview)")
            return

        # Print pandas-style table if available
        if pd is not None and isinstance(preview, pd.DataFrame):
            # indent each printed line
            table_str = preview.to_string()
            for line in table_str.splitlines():
                print(pad + "  " + line)
        else:
            # numpy/list fallback
            try:
                arr = np.asarray(preview)
                # print with limited rows
                if arr.ndim == 0:
                    print(pad + "  ", arr)
                else:
                    for i, row in enumerate(arr):
                        if i >= preview_rows:
                            print(pad + f"  ... ({arr.shape[0]} rows total)")
                            break
                        print(pad + "  ", row)
            except Exception:
                print(pad + "  ", preview)

    elif isinstance(obj, h5py.Group):
        print(f"{pad}+ Group: {name}")
        if obj.attrs:
            print_attrs(obj, indent + 4)


def walk_h5(f):
    # h5py File behaves like a Group
    for name, item in f.items():
        print_h5_item(name, item, indent=0)
        if isinstance(item, h5py.Group):
            # recurse
            for subname, subitem in item.items():
                print_h5_item(f"{name}/{subname}", subitem, indent=4)
                if isinstance(subitem, h5py.Group):
                    for subsubname, subsubitem in subitem.items():
                        print_h5_item(f"{name}/{subname}/{subsubname}", subsubitem, indent=8)


def main(argv):
    if len(argv) < 2:
        print("Usage: python Tests/print_h5_structure.py /path/to/file.h5")
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 3

    try:
        with h5py.File(path, "r") as f:
            print(f"HDF5 file: {path}\n")
            walk_h5(f)
    except Exception as e:
        print(f"Error opening/reading HDF5 file: {e}")
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
