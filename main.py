import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 1. Load and Prepare the Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. XGBoost with GridSearchCV (simplified grid for speed)
param_grid = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.1],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='precision', cv=3, verbose=0, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# 5. Predict probabilities and tune threshold
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# 6. Find threshold with highest precision where recall ≥ 0.6
best_threshold = 0.5  # fallback default
best_precision = 0
min_recall = 0.6

for p, r, t in zip(precisions, recalls, thresholds):
    if r >= min_recall and p > best_precision:
        best_precision = p
        best_threshold = t

print(f"\n🔍 Best Threshold for Precision (Recall ≥ {min_recall}): {best_threshold:.3f}")
print(f"Precision at this threshold: {best_precision:.3f}")

# 7. Predict using improved threshold
y_pred_custom = (y_proba >= best_threshold).astype(int)

# 8. Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred_custom))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# 9. Plot Precision & Recall vs Threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label="Precision", color='blue')
plt.plot(thresholds, recalls[:-1], label="Recall", color='green')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Chosen Threshold = {best_threshold:.2f}')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()
