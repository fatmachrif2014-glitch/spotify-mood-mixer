import pandas as pd

df = pd.read_csv("data/tracks.csv")

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.isna().sum())