import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Load the dataset
data = pd.read_csv('Updated_Social_Bias_Dataset_with_Imputed_Values.csv')


# Define the sensitive features and target variable
sensitive_features = ['Gender', 'Socioeconomic_Status', 'Sexual_Orientation', 'Race/Ethnicity']
target_variable = 'Application_Status'

# Convert the target variable to binary (1 for 'Accepted', 0 for 'Rejected')
data[target_variable] = data[target_variable].apply(lambda x: 1 if x == 'Accepted' else 0)

# Convert all categorical features in the dataset to numerical values using Label Encoding
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Function to create a BinaryLabelDataset for AIF360
def create_binary_label_dataset(data, sensitive_feature, target_variable):
    # Create the BinaryLabelDataset
    binary_dataset = BinaryLabelDataset(
        df=data,
        label_names=[target_variable],
        protected_attribute_names=[sensitive_feature]
    )
    return binary_dataset

# Function to compute fairness metrics using AIF360
def compute_aif360_metrics(data, sensitive_feature, target_variable):
    # Create a BinaryLabelDataset
    binary_dataset = create_binary_label_dataset(data, sensitive_feature, target_variable)
    
    # Define unprivileged and privileged groups
    unprivileged_groups = [{sensitive_feature: min(data[sensitive_feature])}]  # Adjust based on encoded values
    privileged_groups = [{sensitive_feature: max(data[sensitive_feature])}]    # Adjust based on encoded values
    
    # Calculate the fairness metrics
    metric = ClassificationMetric(
        binary_dataset, binary_dataset,  # Using the same dataset for illustration
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Output fairness metrics
    disparate_impact = metric.disparate_impact()
    equal_opportunity_difference = metric.equal_opportunity_difference()
    
    # Return the metrics
    return {
        "Disparate Impact": disparate_impact,
        "Equal Opportunity Difference": equal_opportunity_difference
    }

# Calculate and print the metrics for each sensitive feature
for feature in sensitive_features:
    print(f"AIF360 metrics for {feature}:")
    fairness_metrics = compute_aif360_metrics(data, feature, target_variable)
    print(fairness_metrics)
    print("\n")
