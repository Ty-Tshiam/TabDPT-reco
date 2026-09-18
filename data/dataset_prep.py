# python dataset_prep.py
# python pyspark_pipeline.py >> output.log 2>&1
# python dataset_prep.py >> output.log 2>&1

import os
import json
import polars as pl
import numpy

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(5)

def get_categorical_dictionary(cats, df):
    cat_dict = {}

    for cat in cats:
        #dtype = set(df.select(pl.col(cat)).collect().dtypes)[0]
        #if dtype
        order = df.select(pl.col(cat), pl.col("customer_id"), pl.col("snapshot_date")).group_by(cat).len() .sort("len", descending = True).collect()
        cat_dict[cat] = {}
        for i in range(len(order)):
            cat_dict[cat][order[i, 0]] = i
    
    return cat_dict

def 
path = os.path.join("output", "features", "*.parquet")
df = pl.scan_parquet(path)

non_x =  ["customer_id", "snapshot_date", "target_class"]

# Use schema dict: {col_name: DataType}
schema = df.collect_schema()

x_cats = []

# Columns known to be categorical even if stored as numbers
explicit_cats = {"province_code", "customer_relation_primary"}

for col, dtype in schema.items():
    if col in non_x:
        continue
    # 1. String / Categorical columns
    if dtype in (pl.String, pl.Categorical) or col in explicit_cats:
        x_cats.append(col)

mappings = get_categorical_dictionary(x_cats, df)

mappings['customer_relation_primary'] = {1: 0, 99: 1, 0: 2}

# 2. Apply mappings with default fallbacks
exprs = []
for col, mapping in mappings.items():
    # Determine safe default index (use 'UNKNOWN' or 'OTHER' if available, else  max_idx + 1)
    default_val = mapping.get('UNKNOWN', mapping.get('OTHER', max(mapping.values()) + 1))
    
    # Match the column type to the dictionary key type
    sample_key = next(iter(mapping.keys()))
    cast_type = pl.Int32 if isinstance(sample_key, int) else pl.String

    exprs.append(
        pl.col(col)
        .cast(cast_type)
        .replace(mapping, default=default_val)
        .cast(pl.Int32)
        .alias(col)
    )

df = df.with_columns(exprs)

cols = df.columns
cols = [c for c in df.columns if c not in non_x]
features = df.drop(non_x).collect().to_numpy().flatten()
n_mean = features.mean()
n_std = features.std()

df = df.with_columns((pl.col(cols) - n_mean) / n_std)

n_map = {
    "mean":n_mean,
    "std":n_std
}


os.makedirs("mappings", exist_ok=True)
cat_path = os.path.join("mappings", "mappings.json")
num_path = os.path.join("mappings", "normalize_map.json")

with open(cat_path, "w") as f:
    json.dump(mappings, f, indent=2)
with open(num_path, "w") as f:
    json.dump(n_map, f, indent = 2)

os.makedirs("output/dataset", exist_ok=True)

path = os.path.join("output", "dataset", "dataset_prep.parquet")
df.sink_parquet(path)

