# python dataset_prep.py
# python pyspark_pipeline.py >> output.log 2>&1
# python dataset_prep.py >> output.log 2>&1

import os
import json
import polars as pl

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(5)

path = os.path.join("output", "features", "*.parquet")
df = pl.scan_parquet(path)

non_x =  ["customer_id", "snapshot_date", "target_class"]

# Use schema dict: {col_name: DataType}
schema = df.collect_schema()

x_cats = []
x_nums = []
x_binary = []  # Optional: track 0/1 flags separately

# Columns known to be categorical even if stored as numbers
explicit_cats = {"province_code", "customer_relation_primary"}

for col, dtype in schema.items():
    if col in non_x:
        continue
    # 1. String / Categorical columns
    if dtype in (pl.String, pl.Categorical) or col in explicit_cats:
        x_cats.append(col)
    # 2. Binary flags (ByteType in Spark -> Int8 in Polars, or prefix conventions)
    elif dtype == pl.Int8 or col.startswith("has_") or col.startswith("is_"):
        x_binary.append(col)
        # If your model treats binary as numeric: x_nums.append(col)
        # If your model treats binary as categorical: x_cats.append(col)
    # 3. Continuous and counts (Float64, Float32, Int32, Int64)
    elif dtype.is_numeric():
        x_nums.append(col)

# If treating binary flags as numeric (common in TabDPT / deep tabular models):
x_nums.extend(x_binary)

#print(f"Categorical features ({len(x_cats)}):", x_cats)
#print(f"Numerical features   ({len(x_nums)}):", x_nums)

def categorical_dictionary(cats, df):
    cat_dict = {}

    for cat in cats:
        #dtype = set(df.select(pl.col(cat)).collect().dtypes)[0]
        #if dtype
        order = df.select(pl.col(cat), pl.col("customer_id"), pl.col("snapshot_date")).group_by(cat).len() .sort("len", descending = True).collect()
        cat_dict[cat] = {}
        for i in range(len(order)):
            cat_dict[cat][order[i, 0]] = i
    
    return cat_dict

mappings = categorical_dictionary(x_cats, df)

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

def get_normalize_dictionary(nums, df):
    num_dict = {}
    for num in nums:
        num_dict[num] = {
            "mean" : df.select(pl.col(num).mean()).collect().item(),
            "std" : df.select(pl.col(num).std()).collect().item()
            }

    return num_dict

mappings_nums = get_normalize_dictionary(x_nums, df)

exprs = []
for num in x_nums:
    mapping = mappings_nums[num]
    exprs.append(
        ((pl.col(num) - mapping["mean"]) / mapping["std"]).alias(num)
    )

df = df.with_columns(exprs)

os.makedirs("mappings", exist_ok=True)
cat_path = os.path.join("mappings", "mappings_cats.json")
num_path = os.path.join("mappings", "mappings_nums.json")

with open(cat_path, "w") as f:
    json.dump(mappings, f, indent=2)
with open(num_path, "w") as f:
    json.dump(mappings_nums, f, indent = 2)

print("printing")

os.makedirs("output/dataset", exist_ok=True)

path = os.path.join("output", "dataset", "dataset_prep.parquet")
df.sink_parquet(path)