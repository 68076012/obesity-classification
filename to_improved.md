# Model Improvement Suggestions

## Current Performance Analysis

### Confusion Matrix Results:
- **Normal**: 117 correct, 15 incorrect
  - Confused with Obesity (13 cases) and Underweight (2 cases)
- **Overweight**: 46 correct, 11 incorrect
  - Confused with Obesity (9 cases) and Normal (2 cases)
- **Obesity**: 108 correct, 10 incorrect
  - Best performing class
  - Confused with Normal (7 cases) and Overweight (3 cases)
- **Underweight**: 12 correct, 3 incorrect
  - Smallest sample size
  - Confused with Normal (3 cases)

## Main Issues Identified:

1. **Class Imbalance** - Underweight has insufficient samples
2. **Boundary Confusion** - Overweight vs Obesity hard to distinguish
3. **Feature Overlap** - Normal vs Obesity confusion

---

## Improvement Strategies

### 1. Advanced SMOTE Techniques

**Current approach:** Using basic SMOTE

**Improvements:**
```python
from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, ADASYN

# BorderlineSMOTE - Focus on boundary cases
pipe = Pipeline([
    ('smote', BorderlineSMOTE(random_state=42, k_neighbors=3)),
    ('clf', RandomForestClassifier(random_state=42))
])

# ADASYN - Adaptive Synthetic Sampling
pipe = Pipeline([
    ('smote', ADASYN(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
```

**Why:** Better handles minority classes and focuses on difficult-to-classify boundary regions

---

### 2. Class Weight Balancing

**Add to hyperparameter tuning:**
```python
param_grid_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20],
    'clf__class_weight': ['balanced', 'balanced_subsample', None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

param_grid_gb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__max_depth': [3, 5, 7],
    'clf__subsample': [0.8, 0.9, 1.0]
}
```

**Why:** Penalizes misclassification of minority classes more heavily

---

### 3. Feature Engineering

**New features to create:**

```python
# BMI-related features
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                             labels=['Under', 'Normal', 'Over', 'Obese'])

# Interaction features
df['Exercise_x_Diet'] = df['Physical_Activity'] * df['Food_Consumption']
df['Screen_x_Exercise'] = df['Screen_Time'] * df['Physical_Activity']
df['Calorie_Ratio'] = df['FCVC'] / (df['NCP'] + 1)

# Polynomial features for key variables
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False,
                          interaction_only=True)
```

**Why:** Better capture non-linear relationships and class boundaries

---

### 4. Advanced Ensemble Methods

**A. Stacking Classifier:**
```python
from sklearn.ensemble import StackingClassifier

base_estimators = [
    ('rf', pipe_rf),
    ('gb', pipe_gb),
    ('lr', pipe_lr)
]

stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
```

**B. XGBoost / CatBoost:**
```python
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

pipe_xgb = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(
        random_state=42,
        scale_pos_weight=compute_scale_pos_weight(y_train)
    ))
])

pipe_cat = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', CatBoostClassifier(
        random_state=42,
        class_weights='Balanced',
        verbose=False
    ))
])
```

**Why:** Often outperform traditional methods, better handle imbalanced data

---

### 5. Hyperparameter Optimization

**Use RandomizedSearchCV for wider search:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'clf__n_estimators': randint(100, 500),
    'clf__max_depth': [3, 5, 7, 10, None],
    'clf__min_samples_split': randint(2, 20),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__max_features': ['sqrt', 'log2', None],
    'clf__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=100,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)
```

**Why:** Explores larger parameter space more efficiently

---

### 6. Cost-Sensitive Learning

```python
from sklearn.utils.class_weight import compute_sample_weight

# Compute sample weights
sample_weights = compute_sample_weight('balanced', y_train)

# Use in model training
grid.fit(X_train, y_train, clf__sample_weight=sample_weights)
```

**Why:** Explicitly penalize misclassification of minority classes

---

### 7. Cross-Validation Strategy

**Stratified K-Fold with more splits:**
```python
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Or use RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
```

**Why:** Better estimate of model performance with imbalanced data

---

### 8. Evaluation Metrics

**Use multiple metrics:**
```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# Per-class metrics
print(classification_report(y_test, y_pred, target_names=class_names))

# Weighted metrics
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Focus on minority class performance
f1_per_class = f1_score(y_test, y_pred, average=None)
```

---

### 9. Data Augmentation

**Generate synthetic samples strategically:**
```python
# Oversample only the minority classes
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy={
    'Underweight': 300,  # Target sample count
    'Overweight': 250
}, random_state=42)
```

---

### 10. Neural Network Approach

**Try deep learning:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# Use class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=100,
          batch_size=32,
          class_weight=dict(enumerate(class_weights)),
          validation_split=0.2)
```

---

## Priority Implementation Order:

1. ✅ **Quick wins** (1-2 hours):
   - Add `class_weight='balanced'` to current models
   - Try BorderlineSMOTE instead of basic SMOTE
   - Expand hyperparameter grid

2. 📊 **Feature engineering** (2-3 hours):
   - Create interaction features
   - Add polynomial features
   - Feature selection

3. 🚀 **Advanced models** (3-4 hours):
   - Implement XGBoost/CatBoost
   - Try Stacking ensemble
   - Neural network approach

4. 🔬 **Fine-tuning** (2-3 hours):
   - RandomizedSearchCV with wider ranges
   - Cost-sensitive learning
   - Advanced cross-validation

---

## Expected Improvements:

- **Underweight F1-score**: +10-15% (from better handling of minority class)
- **Overweight-Obesity separation**: +5-10% (from feature engineering)
- **Overall accuracy**: +3-7% (from ensemble improvements)

---

## Monitoring:

Track these metrics after each improvement:
- Per-class F1 scores (especially Underweight)
- Confusion matrix (focus on Overweight-Obesity boundary)
- Weighted F1 score
- Macro F1 score (treats all classes equally)
