import pandas as pd

dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(dataset.columns)

print(dataset.values[0:6])

