import pandas as pd

df = pd.read_parquet("data/features/inmet_sp_daily_features.parquet")
print(df.columns)