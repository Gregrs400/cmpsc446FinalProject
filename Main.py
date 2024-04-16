import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# These are the different columns in our dataset
# print(dataset.columns)


# First 6 data entries
# print(dataset.values[0:6])


# Unique values (ex. for male category: male, female)
ds_columns = dataset.columns.tolist()
for column in ds_columns:
    print(f"{column} unique values : {dataset[column].unique()}")


# Displays the number of people who churned or didn't in bar graph
plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x="Churn", hue="Churn", data=dataset, palette="Blues", legend=False)
# Adding count labels above the bars
for container in ax.containers:
    ax.bar_label(container, label_type='center')
plt.show()


# Important statistics about senior citizens, tenure, and monthly charges
print(dataset.describe())


# Types of data that we have
print(dataset.dtypes)


# Change TotalCharges to float (instead of object)
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")


# Function to create a histogram
def histogram_plots(data_set, numerical_values, target):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values) / 2)

    fig = plt.figure(figsize=(12, 5 * number_of_rows))

    for index, column in enumerate(numerical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        ax = sns.kdeplot(data_set[column][data_set[target] == "Yes"], fill=True)
        ax = sns.kdeplot(data_set[column][data_set[target] == "No"], fill=True)
        ax.set_title(column)
        ax.legend(["Churn", "No Churn"], loc='upper right')
    plt.savefig("numerical_variables.png", dpi=300)
    return plt.show()


# Histogram for churn based on these categories
customer_account_num = ["tenure", "MonthlyCharges", "TotalCharges"]
histogram_plots(dataset, customer_account_num, "Churn")


# Function to create an outlier boxplot
def outlier_check_boxplot(data_set, numerical_values):
    number_of_columns = 2
    number_of_rows = math.ceil(len(numerical_values) / 2)

    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    for index, column in enumerate(numerical_values, 1):
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)
        ax = sns.boxplot(x=column, data=data_set, palette="Blues")
        ax.set_title(column)
    plt.savefig("Outliers_check.png", dpi=300)
    return plt.show()


# Outlier in these categories
numerical_values = ["tenure", "MonthlyCharges", "TotalCharges"]
outlier_check_boxplot(dataset, numerical_values)