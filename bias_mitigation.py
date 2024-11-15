import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, GridSearch
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score

# Optional: Disable oneDNN optimizations if you prefer strict numerical consistency
# tf.config.experimental.set_option('TF_ENABLE_ONEDNN_OPTS', False)  # Uncomment if needed

# Disable eager execution for TensorFlow
tf.compat.v1.disable_eager_execution()

# Load the dataset
data = pd.read_csv('Updated_Social_Bias_Dataset_with_Imputed_Values.csv')


# Define the target variable and sensitive features
target_variable = 'Application_Status'
sensitive_features = ['Gender', 'Socioeconomic_Status', 'Sexual_Orientation', 'Race/Ethnicity']

# Convert the target variable to binary (1 for 'Accepted', 0 for 'Rejected')
data[target_variable] = data[target_variable].apply(lambda x: 1 if x == 'Accepted' else 0)

# Encode all categorical features using LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the dataset into training and testing sets
X = data.drop(columns=[target_variable])
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 1: Re-sampling Example ---
# Train a Logistic Regression model using re-sampling for bias mitigation
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(solver='liblinear'),
    constraints=DemographicParity()
)
mitigator.fit(X_train, y_train, sensitive_features=X_train['Gender'])

# Make predictions on the test set
y_pred_resampled = mitigator.predict(X_test)

# --- Step 2: Re-weighting Example ---
# Apply GridSearch with Demographic Parity constraints for re-weighting
grid_search = GridSearch(LogisticRegression(solver='liblinear'), constraints=DemographicParity())
grid_search.fit(X_train, y_train, sensitive_features=X_train['Gender'])

# Evaluate all predictors and select the best one based on accuracy
best_predictor = None
best_accuracy = 0

for predictor in grid_search.predictors_:
    y_pred = predictor.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_predictor = predictor

# Make predictions on the test set using the best predictor
y_pred_reweighted = best_predictor.predict(X_test)

# --- Step 3: Adversarial De-biasing Example ---
# Function to create a BinaryLabelDataset for AIF360
def create_binary_label_dataset(X, y, sensitive_feature):
    df = X.copy()
    df.loc[:, target_variable] = y  # Use .loc to avoid chained assignment warning
    binary_dataset = BinaryLabelDataset(
        df=df,
        label_names=[target_variable],
        protected_attribute_names=[sensitive_feature]
    )
    return binary_dataset

# Create BinaryLabelDataset for training and testing
binary_dataset_train = create_binary_label_dataset(X_train, y_train, 'Gender')
binary_dataset_test = create_binary_label_dataset(X_test, y_test, 'Gender')

# Configure the TensorFlow session for adversarial debiasing
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()

# Initialize the AdversarialDebiasing model
adversarial_model = AdversarialDebiasing(
    privileged_groups=[{'Gender': 1}],
    unprivileged_groups=[{'Gender': 0}],
    scope_name='debias',
    sess=sess,
    num_epochs=50,  # You can fine-tune this parameter
    batch_size=128,  # You can adjust the batch size
    debias=True  # Enable debiasing during training
)

# Train the adversarial model
adversarial_model.fit(binary_dataset_train)

# Make predictions using the adversarial model
predictions_adversarial = adversarial_model.predict(binary_dataset_test)

# Evaluate the metrics for the adversarially debiased model
metric = ClassificationMetric(
    binary_dataset_test, predictions_adversarial,
    unprivileged_groups=[{'Gender': 0}],
    privileged_groups=[{'Gender': 1}]
)

disparate_impact = metric.disparate_impact()
equal_opportunity_difference = metric.equal_opportunity_difference()

print(f"Adversarial Debiasing Metrics: Disparate Impact: {disparate_impact}, Equal Opportunity Difference: {equal_opportunity_difference}")
