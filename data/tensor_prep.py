# python tensor_prep.py >> output.log 2>&1
import polars as pl
import torch 
import os

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)

data_path = os.path.join("output", "dataset", "dataset_prep.parquet")
#train_path = os.path.join("output", "tabdpt_train", "*.parquet")
test_path = os.path.join("output", "tabdpt_test_may2016", "*.parquet")
orginal_path = os.path.join("santander-product-recommendation", "train_ver2.csv")

og = pl.scan_csv(orginal_path, schema_overrides={"indrel_1mes": pl.String})

test = pl.scan_parquet(test_path)

df = pl.scan_parquet(data_path)
# train = pl.scan_parquet(train_path)

# print(df.collect().head)

targets = df.filter(pl.col("target_class") != 0)

#print(targets.collect().height)

#print(targets.select(pl.col("customer_id"), pl.col("snapshot_date"), pl.col("target_class")).group_by("target_class").len().collect().sort("target_class"))
#print(targets.collect().columns)



print(df.collect().height)
print(df.filter(pl.col("is_active_lag1") == 0).collect().height)

print(test.collect().height)
print(test.filter(pl.col("is_active") == 0).collect().height)
print(
    og.filter(
        pl.col("ind_actividad_cliente").str.strip_chars().cast(pl.Int32, strict=False) == 0
    )
    .collect()
    .height
)