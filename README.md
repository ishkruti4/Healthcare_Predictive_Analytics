# Healthcare Predictive Analytics: Diabetes Detection

This project predicts whether a patient is at risk for diabetes based on standard clinical measurements taken from the Pima Indians Diabetes Database. It compares predictive models while emphasizing data cleaning (dealing with biological anomalies) and strict ethical data standards.

## Project Overview

Raw medical data is notoriously messy. This project shows how to handle "clinical chaos" by cleaning biologically impossible zero readings and normalizing metrics for high-precision diagnostic analysis.

### Preprocessing and Managing "Clinical Chaos"

- **Addressing Biological Zeros:** The Pima Indians dataset contains zero measurements in critical categories where a value of zero is biologically impossible—such as Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI. Leaving these zeros untouched biases machine learning systems. We resolve this by replacing zero values with the median of their respective columns.

- **Feature Scaling:** Since clinical markers use highly disparate units (pregnancies range from $0$ to $17$, while insulin levels can reach up to $846$), we normalize the metrics using **Z-score standardization**. This prevents larger range values from overpowering smaller ones. The formula used to transform each value is:

$$z = \frac{x - \mu}{\sigma}$$

(Where $\mu$ is the mean of the column and $\sigma$ is the standard deviation)

## Machine Learning Models: Interpretability vs. Complexity

In clinical medicine, understanding *why* a machine learning model makes a decision is just as critical as its overall accuracy.

- **Logistic Regression (Highly Interpretable):** Because it operates on direct weighted inputs, clinicians can easily track how each biological marker influences the risk score.
- **Random Forest (Highly Robust):** Aggregates multiple decision flowcharts to handle complex, non-linear relationships, though it is slightly harder to explain directly (considered a "black-box" model).

### Feature Importance (The Strongest Clinical Signals)

During training, models calculate feature importance coefficients. Plasma Glucose emerges as the single strongest clinical indicator, followed by Body Mass Index (BMI), Number of Pregnancies, and Age.

## Ethical Standards & HIPAA Patient Privacy

Training AI models on private clinical data requires compliance with privacy legislation like HIPAA. To work with patient health records safely, data must be fully de-identified. HIPAA outlines two safe pathways:

### 1. The Safe Harbor Method (Checklist-Based)

This method requires the complete removal of **18 specific identifiers** from patient records under **45 CFR § 164.514(b)(2)(i)**. Key identifiers that must be redacted include:

1. Names
2. Geographic subdivisions smaller than a state
3. All dates directly related to an individual (except year), including birth, admission, and discharge dates
4. Telephone and fax numbers
5. Email addresses
6. Social Security numbers
7. Medical record and account numbers
8. Web URLs and IP addresses
9. Biometric identifiers (finger/voice prints) and full-face photos

- **Advantage:** Low technical complexity; easy to implement and audit.
- **Disadvantage:** Heavily reduces data utility; removing geographic trends and precise dates makes it difficult to track localized disease outbreaks.

### 2. The Expert Determination Method (Scientific-Based)

A qualified statistician applies custom mathematical frameworks to evaluate the uniqueness of the data and mitigates risks to ensure that the probability of patient re-identification is "very small".

**Anonymization Techniques Used:**

- **$k$-Anonymity:** Aggregates quasi-identifiers (such as age or location) so that any individual's record is completely indistinguishable from at least $k-1$ other individuals.
- **Differential Privacy:** Intentionally injects calculated mathematical "noise" into query outputs to protect individual patient identities while preserving broader statistical analysis.

## How to Run the Code

1. Install the required Python packages:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

2. Place your raw dataset file (`diabetes.csv`) in the project folder.

3. Run the Python script:

```bash
python main.py
```
