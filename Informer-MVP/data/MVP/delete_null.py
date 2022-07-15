import pandas as pd
data = pd.read_csv("3x3_Vehicle_Traffic_FLow.csv")
res = data.dropna(how="all")
res.to_csv("3x3_Vehicle_Traffic_FLow_1.csv", index=False)
