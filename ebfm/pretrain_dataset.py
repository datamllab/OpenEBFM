"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for locating and iterating over the parquet files.
Downloading lives in `ebfm/dataset.py` (`python -m ebfm.dataset`).

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os

import pyarrow.parquet as pq

from ebfm.common import get_base_dir


VALID_DATASETS = ("fineweb", "climbmix", "dclm", "owt")
OWT_VALIDATION_DOCS = 100_000
DEFAULT_DATASET = "dclm"


def normalize_dataset(dataset=None):
    """Return a validated pretraining dataset name."""
    dataset = dataset or DEFAULT_DATASET
    dataset = str(dataset).strip().lower()
    if dataset not in VALID_DATASETS:
        valid = ", ".join(VALID_DATASETS)
        raise ValueError(
            f"Unknown dataset={dataset!r}; expected one of: {valid}"
        )
    return dataset


def get_dataset_dir(dataset=None, data_dir=None):
    """
    Return the local parquet directory for a base pretraining dataset.

    The layout is:
      $EBFM_BASE_DIR/fineweb/
      $EBFM_BASE_DIR/climbmix/
      $EBFM_BASE_DIR/dclm/
      $EBFM_BASE_DIR/owt/
    """
    if data_dir:
        return os.path.abspath(data_dir)

    dataset = normalize_dataset(dataset)
    return os.path.join(get_base_dir(), dataset)


def list_parquet_files(data_dir=None, dataset=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = get_dataset_dir(dataset=dataset, data_dir=data_dir)
    if not os.path.isdir(data_dir):
        return []
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def get_owt_split_ranges(parquet_paths, validation_docs=OWT_VALIDATION_DOCS):
    """Return exact document ranges for the OWT train/validation split.

    The MDLM OWT protocol reserves the *last* 100,000 documents, rather than
    reserving the last parquet shard.  The returned list contains one
    ``(path, start_row, end_row)`` tuple per file that contributes to the
    split; row offsets are relative to that parquet file and ``end_row`` is
    exclusive.  Metadata is used, so no document data is read while building
    the split.
    """
    if not parquet_paths:
        return {"train": [], "val": [], "total_docs": 0}

    file_rows = [
        (path, int(pq.ParquetFile(path).metadata.num_rows)) for path in parquet_paths
    ]
    total_docs = sum(num_rows for _, num_rows in file_rows)
    if total_docs < validation_docs:
        raise ValueError(
            f"OWT contains {total_docs} documents, fewer than the requested "
            f"{validation_docs}-document validation split"
        )

    train_end = total_docs - validation_docs
    result = {"train": [], "val": [], "total_docs": total_docs}
    corpus_start = 0
    for path, num_rows in file_rows:
        corpus_end = corpus_start + num_rows
        train_start = 0
        train_stop = max(0, min(num_rows, train_end - corpus_start))
        val_start = max(0, min(num_rows, train_end - corpus_start))
        val_stop = num_rows
        if train_stop > train_start:
            result["train"].append((path, train_start, train_stop))
        if val_stop > val_start:
            result["val"].append((path, val_start, val_stop))
        corpus_start = corpus_end
    return result
