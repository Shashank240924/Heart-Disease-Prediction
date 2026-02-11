import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek  # Advanced imbalance handling

# 1. Load and Clean Data
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(columns=['education'], inplace=True)
disease_df.rename(columns={'male':'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

# 2. Feature Selection
# Using broader feature set as medical outcomes often depend on subtle interactions
features = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'prevalentHyp', 'diabetes']
X = disease_df[features]
y = disease_df['TenYearCHD']

# 3. Split & Scale (Scale only after split to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Handle Imbalance with SMOTETomek
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train_scaled, y_train)

# 5. Model: Random Forest with Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_resampled, y_resampled)
best_model = grid_search.best_estimator_

# 6. Evaluation
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"Best Parameters: {grid_search.best_params_}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualizing Performance
# Confusion Matrix
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()