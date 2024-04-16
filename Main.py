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
        ax = sns.boxplot(x=column, hue=0, data=data_set, palette="Blues", legend=False)
        ax.set_title(column)
    plt.savefig("Outliers_check.png", dpi=300)
    return plt.show()


# Outlier in these categories
numerical_values = ["tenure", "MonthlyCharges", "TotalCharges"]
outlier_check_boxplot(dataset, numerical_values)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='MultipleLines', hue='Churn', data=dataset, palette='Blues', legend=True)
plt.savefig("churnGraphs\\MultipleLinesChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='gender', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\genderChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='SeniorCitizen', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\SeniorCitizenChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='Partner', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\PartnerChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='Dependents', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\DependentsChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='PhoneService', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\PhoneServiceChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='InternetService', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\InternetServiceChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='OnlineSecurity', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\OnlineSecurityChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='OnlineBackup', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\OnlineBackupChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='DeviceProtection', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\DeviceProtectionChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='TechSupport', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\TechSupportChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='StreamingTV', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\StreamingTVChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='StreamingMovies', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\StreamingMoviesChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='Contract', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\ContractChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='PaperlessBilling', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\PaperlessBillingChurn.png", dpi=300)

plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x='PaymentMethod', hue='Churn', data=dataset, palette="Blues", legend=True)
plt.savefig("churnGraphs\\PaymentMethodChurn.png", dpi=300)