import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# print(dataset.columns)

# print(dataset.values[0:6])

# ds_columns = dataset.columns.tolist()
# for column in ds_columns:
#     print(f"{column} unique values : {dataset[column].unique()}")


print(dataset.columns.values)


# plt.style.use("ggplot")
# plt.figure(figsize=(5, 5))
# ax = sns.countplot(x="Churn", hue="Churn", data=dataset, palette="Blues", legend=False)
#
# # Adding count labels above the bars
# for container in ax.containers:
#     ax.bar_label(container, label_type='center')
#
# # plt.show()
#
#
# # print(dataset.columns[5])
#
# print(dataset.describe())
#
# print(dataset.dtypes)
#
# # Change TotalCharges to float
# dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")
#
#
# def histogram_plots(df, numerical_values, target):
#     number_of_columns = 2
#     number_of_rows = math.ceil(len(numerical_values) / 2)
#
#     fig = plt.figure(figsize=(12, 5 * number_of_rows))
#
#     for index, column in enumerate(numerical_values, 1):
#         ax = fig.add_subplot(number_of_rows, number_of_columns, index)
#         ax = sns.kdeplot(df[column][df[target] == "Yes"], fill=True)
#         ax = sns.kdeplot(df[column][df[target] == "No"], fill=True)
#         ax.set_title(column)
#         ax.legend(["Churn", "No Churn"], loc='upper right')
#     plt.savefig("numerical_variables.png", dpi=300)
#     return plt.show()
#
#
# customer_account_num = ["tenure", "MonthlyCharges", "TotalCharges"]
# histogram_plots(dataset, customer_account_num, "Churn")
#
#
# def outlier_check_boxplot(df, numerical_values):
#     number_of_columns = 2
#     number_of_rows = math.ceil(len(numerical_values) / 2)
#
#     fig = plt.figure(figsize=(12, 5 * number_of_rows))
#     for index, column in enumerate(numerical_values, 1):
#         ax = fig.add_subplot(number_of_rows, number_of_columns, index)
#         ax = sns.boxplot(x=column, data=df, palette="Blues")
#         ax.set_title(column)
#     plt.savefig("Outliers_check.png", dpi=300)
#     return plt.show()
#
#
# numerical_values = ["tenure", "MonthlyCharges", "TotalCharges"]
# outlier_check_boxplot(dataset, numerical_values)
#
# # bar graph for multiple lines and churn
#
# plt.style.use("ggplot")
# plt.figure(figsize=(5, 5))
# ax = sns.countplot(x="MultipleLines", hue='Churn', data=dataset, palette="Blues", legend=True)
# plt.show()
#
# plt.style.use("ggplot")
# plt.figure(figsize=(5, 5))
# ax = sns.countplot(x="MultipleLines", hue='Churn', data=dataset, palette="Blues", legend=True)
# plt.show()

