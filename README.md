
# Addressing Social Bias in Datasets

## Overview
This project focuses on developing a comprehensive solution to identify, mitigate, and reduce inherent biases in datasets related to race, ethnicity, gender, sexuality, religion, socioeconomic status, and other social attributes. The project includes bias analysis, detection, and mitigation using state-of-the-art techniques and libraries.

## Project Structure
The project is divided into four main sections:

### 1. **Bias Analysis**
- **Script**: `Bias_Analysis.py`
- **Description**: Loads the dataset, conducts exploratory data analysis (EDA), and computes fairness metrics such as disparate impact, equal opportunity difference, and statistical parity difference.
- **Key Functions**:
  - `calculate_disparate_impact()`
  - `calculate_equal_opportunity_difference()`
  - `calculate_statistical_parity_difference()`
- **Output**: Displays metrics for sensitive features like `Gender`, `Socioeconomic_Status`, `Sexual_Orientation`, and `Race/Ethnicity`.

### 2. **Bias Detection using Fairlearn**
- **Script**: `Bias_Detection.py`
- **Description**: Utilizes the `fairlearn` library to compute fairness metrics, including demographic parity difference and equalized odds difference for each sensitive attribute.
- **Key Libraries**:
  - `fairlearn`, `pandas`, `sklearn`
- **Output**: Provides fairness metrics for sensitive features in the dataset.

### 3. **Bias Detection using AIF360**
- **Script**: `Bias_Detection_AIF360.py`
- **Description**: Uses `AIF360`, a toolkit for measuring fairness in machine learning models. This script encodes categorical data, creates a `BinaryLabelDataset`, and calculates fairness metrics like disparate impact and equal opportunity difference.
- **Key Libraries**:
  - `aif360`, `pandas`, `sklearn.preprocessing`
- **Key Functions**:
  - `create_binary_label_dataset()`
  - `compute_aif360_metrics()`
- **Output**: Detailed metrics using AIF360 for various sensitive features.

### 4. **Bias Mitigation**
- **Script**: `Bias_Mitigation.py`
- **Description**: Implements bias mitigation techniques, including re-sampling, re-weighting, and adversarial debiasing using `fairlearn`, `aif360`, and TensorFlow.
- **Key Techniques**:
  - **Re-sampling**: Uses `ExponentiatedGradient` with `DemographicParity` constraints.
  - **Re-weighting**: Applies `GridSearch` to find the best model under fairness constraints.
  - **Adversarial Debiasing**: Employs the `AdversarialDebiasing` algorithm from `AIF360`.
- **Key Libraries**:
  - `tensorflow`, `fairlearn`, `aif360`, `pandas`, `sklearn`
- **Output**: Mitigated model metrics, such as disparate impact and equal opportunity difference.

## Dataset
The project uses the `Expanded_Social_Bias_Dataset`, which includes:
- **Applicant Demographics**: Age, Gender, Race/Ethnicity, Sexual Orientation, Religion, Marital Status, Socioeconomic Status, Education Level.
- **Professional Details**: Years of Experience, Industry, Job Position Applied For.
- **Application Outcomes**: Application Status, Salary Offer, Interview Feedback Score, Referral Status.

This dataset helps in exploring employment trends and potential social biases in the job application process.

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tan2810/Addressing-Social-Bias.git
   cd Addressing-Social-Bias
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure required libraries such as `fairlearn`, `aif360`, `tensorflow`, `pandas`, and `scikit-learn` are included.

3. **Run the scripts**:
   ```bash
   python Bias_Analysis.py
   python Bias_Detection.py
   python Bias_Detection_AIF360.py
   python Bias_Mitigation.py
   ```

## Usage
- **Load the dataset**: Ensure that the dataset `Updated_Social_Bias_Dataset_with_Imputed_Values.csv` is in the project directory.
- **Modify code**: Adjust sensitive features or target variables as needed for your analysis.

## Results
Each script provides bias analysis and fairness metrics. The mitigation script outputs metrics that highlight bias reduction through various techniques.

## Future Work
- **Enhance bias mitigation strategies**: Incorporate more advanced debiasing algorithms.
- **Model training and evaluation**: Extend to include training and testing with real predictions for comprehensive evaluation.
- **Generalize to other datasets**: Apply the methodology to different datasets for broader applicability.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or collaborations, please contact [Your Name](mailto:tanmaybandaru@gmail.com).
