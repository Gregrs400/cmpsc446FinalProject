import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(dataset.columns)

print(dataset.values[0:6])

#Apply the ggplot style
plt.style.use("ggplot")
plt.figure(figsize=(5,5))
ax = sns.countplot(x = dataset["Churn"],palette="Blues")
ax.bar_label(ax.containers[0])
plt.show()