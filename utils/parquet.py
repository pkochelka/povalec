import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_chunked(df, path, chunk_size=100_000):
    """Write `df` to `path` in row-group chunks, so a multi-GB frame never has to be
    materialised as one arrow table."""
    writer = None
    try:
        for start in range(0, len(df), chunk_size):
            table = pa.Table.from_pandas(df.iloc[start:start + chunk_size], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
