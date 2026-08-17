"""Stream the Kaggle CSV files to Parquet without loading an entire file.

Run from the repository root, for example:
    python scripts/csv_to_parquet.py

The invoice source contains a mixed-type ``counter_statue`` column (values such
as 0 and A), so its type is declared as string rather than inferred from the
first CSV block.

The motivation was to save the file. I needed the script as the kernel crashes during web command
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq


CLIENT_TYPES = {
    "disrict": pa.int16(),
    "client_id": pa.string(),
    "client_catg": pa.int16(),
    "region": pa.int16(),
    "creation_date": pa.string(),
    "target": pa.float32(),
}

CLIENT_TEST_TYPES = {key: value for key, value in CLIENT_TYPES.items() if key != "target"}

INVOICE_TYPES = {
    "client_id": pa.string(),
    "invoice_date": pa.string(),
    "tarif_type": pa.int16(),
    "counter_number": pa.int64(),
    "counter_statue": pa.string(),
    "counter_code": pa.int16(),
    "reading_remarque": pa.int16(),
    "counter_coefficient": pa.int16(),
    "consommation_level_1": pa.int64(),
    "consommation_level_2": pa.int64(),
    "consommation_level_3": pa.int64(),
    "consommation_level_4": pa.int64(),
    "old_index": pa.int64(),
    "new_index": pa.int64(),
    # The source mostly contains small month counts but has a large outlier
    # (83125), so int16 is unsafe here.
    "months_number": pa.int64(),
    "counter_type": pa.string(),
}

FILE_TYPES = {
    "client_train": CLIENT_TYPES,
    "client_test": CLIENT_TEST_TYPES,
    "invoice_train": INVOICE_TYPES,
    "invoice_test": INVOICE_TYPES,
}


def convert_one(input_path: Path, output_path: Path, column_types: dict[str, pa.DataType]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    read_options = pv.ReadOptions(block_size=1 << 20, use_threads=True)
    convert_options = pv.ConvertOptions(column_types=column_types, strings_can_be_null=True)
    reader = pv.open_csv(input_path, read_options=read_options, convert_options=convert_options)

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=100_000)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    print(f"{input_path} -> {output_path} ({row_count:,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument(
        "--files",
        nargs="+",
        default=["client_train", "client_test", "invoice_train", "invoice_test"],
        help="File stems without .csv (defaults to all four data tables)",
    )
    args = parser.parse_args()

    for stem in args.files:
        if stem not in FILE_TYPES:
            raise SystemExit(f"Unsupported file stem {stem!r}; choose from {sorted(FILE_TYPES)}")
        input_path = args.data_dir / f"{stem}.csv"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        convert_one(input_path, args.output_dir / f"{stem}.parquet", FILE_TYPES[stem])


if __name__ == "__main__":
    main()
