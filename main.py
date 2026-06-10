import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# Set random seed for reproducibility
np.random.seed(42)

def load_or_generate_data():
    csv_filename = "diabetes.csv"
    if os.path.exists(csv_filename):
        print(f"--- Found real dataset: '{csv_filename}'. Loading... ---")
        df = pd.read_csv(csv_filename)
    else:
        print(f"--- Real dataset '{csv_filename}' not found. Generating clinical synthetic data... ---")
        # Generate 500 patient records following Pima Indians clinical statistics 
        n_samples = 500
        df = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 15, n_samples),
            'Glucose': np.random.choice([0, np.random.uniform(70, 199)], n_samples, p=[0.05, 0.95]), # some zeros
            'BloodPressure': np.random.choice([0, np.random.uniform(50, 120)], n_samples, p=[0.05, 0.95]),
            'SkinThickness': np.random.choice([0, np.random.uniform(10, 50)], n_samples, p=[0.2, 0.8]),
            'Insulin': np.random.choice([0, np.random.uniform(15, 300)], n_samples, p=[0.35, 0.65]),
            'BMI': np.random.choice([0, np.random.uniform(18, 55)], n_samples, p=[0.05, 0.95]),
            'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
            'Age': np.random.randint(21, 80, n_samples),
            'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]) # Standard 34.9% diabetic ratio [24]
        })
        # Round clinical variables to integers where appropriate
        df['Glucose'] = df['Glucose'].astype(int)
        df['BloodPressure'] = df['BloodPressure'].astype(int)
        df['SkinThickness'] = df['SkinThickness'].astype(int)
        df['Insulin'] = df['Insulin'].astype(int)
    return df

# 1. Load data 
df = load_or_generate_data()

# 2. Data Cleaning: Handle Clinical Biological Zeros 
print("Cleaning 'clinical chaos' (resolving biological zeros with column medians)...")
biological_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in biological_zero_cols:
    # Calculate median from non-zero entries to avoid biasing the values 
    median_val = df.loc[df[col] > 0, col].median()
    df[col] = df[col].replace(0, median_val)

# Separate predictors from labels
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# 3. Z-score Standardization 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dataset Split complete. Training records: {X_train.shape} | Test records: {X_test.shape}")

# 5. Initialize Models 
# Clinicians prefer models with high explainability, so we focus on Logistic Regression 
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

# 6. Train and Evaluate Models 
results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall (Sensitivity)': recall_score(y_test, predictions),
        'ROC-AUC Score': roc_auc_score(y_test, probabilities)
    }
    
    print(classification_report(y_test, predictions))

# Display Comparative Metrics
results_df = pd.DataFrame(results).T
print("\n=== Model Evaluation Summary ===")
print(results_df.to_string())

# 7. Extract Feature Importance for Logistic Regression (Clinically Interpretable) 
lr_coefficients = models['Logistic Regression'].coef_[0]
importance_df = pd.DataFrame({
    'Clinical Marker': X.columns,
    'Regression Coefficient': lr_coefficients
})
importance_df = importance_df.sort_values(by='Regression Coefficient', ascending=False)

# Plot Clinical Coefficients
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Regression Coefficient', y='Clinical Marker', palette='coolwarm')
plt.title('Clinical Risk Influencers (Logistic Regression Coefficients)')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('clinical_feature_coefficients.png')
print("\nClinical coefficients plot exported to 'clinical_feature_coefficients.png'")
plt.close()