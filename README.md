# TabDPT-reco: Santander Product Recommendation Pipeline

Data processing and feature engineering pipeline for tabular deep learning models (such as TabDPT) on the Santander Product Recommendation dataset.

## Structure
- `data/pyspark_pipeline.py`: End-to-end PySpark pipeline for cleaning, lag feature engineering, categorical capping (<=100 categories), and 16-class target construction (0 = do nothing, 1..15 = product additions).
- `data/santander-product-recommendation/`: Raw train and test CSVs (ignored by git due to file size).
- `data/output/`: Pre-cleaned and partitioned Parquet datasets (ignored by git).

## Setup
```bash
pip install -r requirements.txt
```
If using PySpark on Linux, ensure Java JRE is installed (`apt install default-jre`).
