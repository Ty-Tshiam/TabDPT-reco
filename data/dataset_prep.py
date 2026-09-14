import polars as pl
import os

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(5)

train_path = os.path.join("output", "tabdpt_train", "*.parquet")

df = pl.scan_parquet(train_path)

col_types = dict(zip(df.columns, df.dtypes))