import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Updated_Social_Bias_Dataset_with_Imputed_Values.csv')

# Inspect the data
print(data.head())
print(data.describe())
print(data.info())

#Phase1
#Defining Metrics

def calculate_disparate_impact(data, group_col, outcome_col, positive_outcome='Accepted'):
    group_1_rate = data[data[group_col] == data[group_col].unique()[0]][outcome_col].value_counts(normalize=True).get(positive_outcome, 0)
    group_2_rate = data[data[group_col] == data[group_col].unique()[1]][outcome_col].value_counts(normalize=True).get(positive_outcome, 0)
    return group_1_rate / group_2_rate if group_2_rate > 0 else None

# Example usage
disparate_impact_gender = calculate_disparate_impact(data, 'Gender', 'Application_Status')
print(f'Disparate Impact (Gender): {disparate_impact_gender}')

def calculate_equal_opportunity_difference(data, group_col, outcome_col, true_positive_label='Accepted'):
    group_1_positive_rate = data[(data[group_col] == data[group_col].unique()[0]) & (data[outcome_col] == true_positive_label)].shape[0] / data[data[group_col] == data[group_col].unique()[0]].shape[0]
    group_2_positive_rate = data[(data[group_col] == data[group_col].unique()[1]) & (data[outcome_col] == true_positive_label)].shape[0] / data[data[group_col] == data[group_col].unique()[1]].shape[0]
    return group_1_positive_rate - group_2_positive_rate

# Example usage
equal_opportunity_diff_gender = calculate_equal_opportunity_difference(data, 'Gender', 'Application_Status')
print(f'Equal Opportunity Difference (Gender): {equal_opportunity_diff_gender}')

def calculate_statistical_parity_difference(data, group_col, outcome_col, positive_outcome='Accepted'):
    group_1_rate = data[data[group_col] == data[group_col].unique()[0]][outcome_col].value_counts(normalize=True).get(positive_outcome, 0)
    group_2_rate = data[data[group_col] == data[group_col].unique()[1]][outcome_col].value_counts(normalize=True).get(positive_outcome, 0)
    return group_1_rate - group_2_rate

# Example usage
statistical_parity_diff_gender = calculate_statistical_parity_difference(data, 'Gender', 'Application_Status')
print(f'Statistical Parity Difference (Gender): {statistical_parity_diff_gender}')

statistical_parity_diff_gender = calculate_statistical_parity_difference(data, 'Socioeconomic_Status', 'Application_Status')
print(f'Statistical Parity Difference (Socioeconomic_Status): {statistical_parity_diff_gender}')

statistical_parity_diff_gender = calculate_statistical_parity_difference(data, 'Sexual_Orientation', 'Application_Status')
print(f'Statistical Parity Difference (Sexual_Orientation): {statistical_parity_diff_gender}')

statistical_parity_diff_gender = calculate_statistical_parity_difference(data, 'Race/Ethnicity', 'Application_Status')
print(f'Statistical Parity Difference (Race/Ethnicity): {statistical_parity_diff_gender}')

