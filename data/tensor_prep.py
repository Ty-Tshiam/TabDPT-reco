# python tensor_prep.py >> output.log 2>&1
import polars as pl
import torch 
import os
import numpy as np

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)

data_path = os.path.join("output", "dataset", "dataset_prep.parquet")

df = pl.scan_parquet(data_path)

#print(df.select(pl.col("target_class", "snapshot_date", "customer_id")).group_by("target_class").len().sort("len").collect())

# Compute max date once, reuse everywhere
max_date = df.select(pl.col("snapshot_date").max()).collect().item()
cutoff = pl.lit(max_date).dt.offset_by("-6mo")

def safe_sample(g, n, seed=42):
    return g.sample(n=min(n, g.height), seed=seed)

# 60k product purchases, stratified by target_class
# 45k recent product purchases from the last 6 months
rp = (
    df.filter(
        (pl.col("target_class") != 0)
        & (pl.col("snapshot_date") >= cutoff)
    )
    .collect()
    .group_by("target_class")
    .map_groups(lambda g: safe_sample(g, 3000, 42))
)

# 15k historical product purchases
hp = (
    df.filter(
        (pl.col("target_class") != 0)
        & (pl.col("snapshot_date") < cutoff)
    )
    .collect()
    .group_by("target_class")
    .map_groups(lambda g: safe_sample(g, 1000, 42))
)

# 30k non purchases
# 21k recent non purchases (active)
rn = (
    df.filter(
        (pl.col("target_class") == 0)
        & (pl.col("snapshot_date") >= cutoff)
        & (pl.col("is_active_lag1") == pl.col("is_active_lag1").max())
    )
    .collect()
    .pipe(lambda d: safe_sample(d, 21000, 42))
)

# 9k historical non purchases (active)
hn = (
    df.filter(
        (pl.col("target_class") == 0)
        & (pl.col("snapshot_date") < cutoff)
        & (pl.col("is_active_lag1") == pl.col("is_active_lag1").max())
    )
    .collect()
    .pipe(lambda d: safe_sample(d, 9000, 42))
)

# 10k non-active non purchases
# 7k recent non-active non purchases
rnn = (
    df.filter(
        (pl.col("target_class") == 0)
        & (pl.col("snapshot_date") >= cutoff)
        & (pl.col("is_active_lag1") == pl.col("is_active_lag1").min())
    )
    .collect()
    .pipe(lambda d: safe_sample(d, 7000, 42))
)

# 3k historical non-active non purchases
hnn = (
    df.filter(
        (pl.col("target_class") == 0)
        & (pl.col("snapshot_date") < cutoff)
        & (pl.col("is_active_lag1") == pl.col("is_active_lag1").min())
    )
    .collect()
    .pipe(lambda d: safe_sample(d, 3000, 42))
)

# Combine everything: 70% recent / 30% historical, 100k total
final_sample = pl.concat([rp, hp, rn, hn, rnn, hnn])

non_x = ["target_class", "snapshot_date", "customer_id"]
xs = [c for c in df.columns if c not in non_x]
final_sample = final_sample.with_columns(pl.col(xs).clip(-10,10))

#print(final_sample.height)  # sanity check -> should be 100_000 (or less, if any group got clipped by safe_sample)

#print(final_sample.head(5))

#print(final_sample.select(pl.col("target_class", "snapshot_date", "customer_id")).group_by("target_class").len().sort("len"))

y = torch.tensor(final_sample["target_class"].to_numpy(), dtype = torch.long)
x = final_sample.drop(pl.col(non_x)).to_numpy()

rows, columns = x.shape

pads = np.zeros((rows, 128 - columns), dtype = np.float32)
matrix = np.hstack([x, pads])

context = torch.tensor(matrix, dtype = torch.bfloat16) # DOUBLE CHECK GPU CONFIGURATION

os.makedirs("tensors", exist_ok=True)

tensor_path = os.path.join("tensors", "context.pt")
y_path = os.path.join("tensors","y.pt")
torch.save(context, tensor_path)
torch.save(y, y_path)

print(context.shape)