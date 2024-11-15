import pandas as pd
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Updated_Social_Bias_Dataset_with_Imputed_Values.csv')

# Define the sensitive features and target variable
sensitive_features = ['Gender', 'Socioeconomic_Status', 'Sexual_Orientation', 'Race/Ethnicity']
target_variable = 'Application_Status'

# Convert the target variable to binary (1 for 'Accepted', 0 for 'Rejected')
data[target_variable] = data[target_variable].apply(lambda x: 1 if x == 'Accepted' else 0)

# Define a function to calculate fairness metrics for each sensitive feature
def compute_fairness_metrics(data, sensitive_feature, target_variable):
    # Calculate the demographic parity difference
    dpd = demographic_parity_difference(
        y_true=data[target_variable],
        y_pred=data[target_variable],  # Replace with model predictions later
        sensitive_features=data[sensitive_feature]
    )

    # Calculate the equalized odds difference
    eod = equalized_odds_difference(
        y_true=data[target_variable],
        y_pred=data[target_variable],  # Replace with model predictions later
        sensitive_features=data[sensitive_feature]
    )

    # Return the calculated metrics
    return {
        "Demographic Parity Difference": dpd,
        "Equalized Odds Difference": eod
    }

# Calculate fairness metrics for each sensitive feature
for feature in sensitive_features:
    print(f"Fairness metrics for {feature}:")
    fairness_metrics = compute_fairness_metrics(data, feature, target_variable)
    print(fairness_metrics)
    print("\n")
