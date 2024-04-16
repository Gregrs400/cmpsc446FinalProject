import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Types of data that we have (identify the columns that we have and their types)
print(dataset.dtypes)

# Unique values for categorical columns
ds_columns = dataset.columns.tolist()
for column in ds_columns:
    if dataset[column].dtypes != 'int64' and dataset[column].dtypes != 'float64':
        print(f'{column} : {dataset[column].unique()}')

# Remove "(automatic)" from "PaymentMethod" column
dataset["PaymentMethod"] = dataset["PaymentMethod"].str.replace(" (automatic)", "", regex=False)

# Bar graph showing the number of churned and non-churned customers
plt.style.use("ggplot")
plt.figure(figsize=(5, 5))
ax = sns.countplot(x="Churn", hue="Churn", data=dataset, palette="Blues", legend=False)
for container in ax.containers:
    ax.bar_label(container, label_type='center')
plt.savefig("churnGraphs\\churnedvsnot.png", dpi=300)

# Important statistics about senior citizens, tenure, and monthly charges
print(dataset.describe())

# Change "TotalCharges" to float
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

# Check for null values
print(dataset.isnull().sum())

# Remove rows with null values
dataset.dropna(inplace=True)

# Remove customer id column
dataset = dataset.iloc[:, 1:]


# Histograms for numerical columns
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


numerical_values = ["tenure", "MonthlyCharges", "TotalCharges"]
histogram_plots(dataset, numerical_values, "Churn")


# Box plots to check for outliers
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


numerical_values = ["tenure", "MonthlyCharges", "TotalCharges"]
outlier_check_boxplot(dataset, numerical_values)


# Correlation of "Churn" with other variables
dataset['Churn'].replace(to_replace='Yes', value=1, inplace=True)
dataset['Churn'].replace(to_replace='No', value=0, inplace=True)
dataset_dummies = pd.get_dummies(dataset)
plt.figure(figsize=(15, 8))
dataset_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.savefig("churnCorrelation.png", dpi=300)


# ------------ Graphs showing Churn rates for different useful attributes ------------
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


# ------------------------ Machine Learning Phase (testing and training) ------------------------

# Identify categorical columns
categorical_cols = dataset.select_dtypes(include=['category', 'object']).columns

# Apply OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(dataset[categorical_cols])
encoded_dataset = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
dataset.drop(columns=categorical_cols, inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset = pd.concat([dataset, encoded_dataset], axis=1)

# Drop 'Churn_No' column if it exists
if 'Churn_No' in dataset.columns:
    dataset.drop('Churn_No', axis=1, inplace=True)

# Rename 'Churn_Yes' to 'Churn'
dataset.rename(columns={'Churn_Yes': 'Churn'}, inplace=True)

# Split data into X and y
X = dataset.drop('Churn', axis=1)
y = dataset['Churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Define functions for evaluation
def feature_weights(X_df, classifier, classifier_name):
    weights = pd.Series(classifier.coef_[0], index=X_df.columns.values).sort_values(ascending=False)
    top_10_weights = weights[:10]
    plt.figure(figsize=(7, 6))
    plt.title(f"{classifier_name} - Top 10 Features")
    top_10_weights.plot(kind="bar")

    bottom_10_weights = weights[len(weights) - 10:]
    plt.figure(figsize=(7, 6))
    plt.title(f"{classifier_name} - Bottom 10 Features")
    bottom_10_weights.plot(kind="bar")
    print("")


def confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred, classifier, classifier_name):
    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.show()
    print(f"Accuracy Score Test = {accuracy_score(y_pred, y_test)}")
    print(f"Accuracy Score Train = {classifier.score(X_train, y_train)}")
    return print("\n")


def roc_curve_auc_score(X_test, y_test, y_pred_probabilities, classifier_name):
    y_pred_prob = y_pred_probabilities[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"{classifier_name}")
    plt.title(f"{classifier_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    return print(f"AUC Score (ROC):{roc_auc_score(y_test, y_pred_prob)}")


def precision_recall_curve_and_scores(X_test, y_test, y_pred, y_pred_probabilities, classifier_name):
    y_pred_prob = y_pred_probabilities[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    plt.plot(recall, precision, label=f"{classifier_name}")
    plt.title(f"{classifier_name}-PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()
    f1_score_result, auc_score = f1_score(y_test, y_pred), auc(recall, precision)
    return print(f"f1 Score : {f1_score_result} \n AUC Score (PR) : {auc_score}")


# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_knn_proba = knn.predict_proba(X_test)
print("\n\n\n---------- Results for K-nearest neighbors ----------")
confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred_knn, knn, "K-Nearest Neighbors")
roc_curve_auc_score(X_test, y_test, y_pred_knn_proba, "K-Nearest Neighbors")
precision_recall_curve_and_scores(X_test, y_test, y_pred_knn, y_pred_knn_proba, "K-Nearest Neighbors")

# Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_pred_logreg_proba = logreg.predict_proba(X_test)
feature_weights(X_train, logreg, "Logistic Regression")
print("\n---------- Results for Logistic Regression ----------")
confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred_logreg, logreg, "Logistic Regression")
roc_curve_auc_score(X_test, y_test, y_pred_logreg_proba, "Logistic Regression")
precision_recall_curve_and_scores(X_test, y_test, y_pred_knn, y_pred_logreg_proba, "Logistic Regression")

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
y_pred_decision_tree_proba = decision_tree.predict_proba(X_test)
print("\n---------- Results for Decision Tree ----------")
confusion_matrix_plot(X_train, y_train, X_test, y_test, y_pred_decision_tree, decision_tree, "Decision Tree")
roc_curve_auc_score(X_test, y_test, y_pred_decision_tree_proba, "Decision Tree")
precision_recall_curve_and_scores(X_test, y_test, y_pred_decision_tree, y_pred_decision_tree_proba, "Decision Tree")

