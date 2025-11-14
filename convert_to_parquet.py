import pandas as pd

print("Loading 245MB CSV…")
df = pd.read_csv("HR_Data.csv")

print("Saving compressed Parquet…")
df.to_parquet("HR_Data.parquet", compression="snappy")

print("Done! HR_Data.parquet created successfully.")
