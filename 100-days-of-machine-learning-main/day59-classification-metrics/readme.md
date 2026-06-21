# Day 59 — Classification Metrics

## Why Classification Metrics Matter

Accuracy is misleading for imbalanced datasets. If 95% of emails are not spam, a model that predicts "not spam" for everything achieves 95% accuracy — yet it is completely useless. Classification metrics beyond accuracy capture what actually matters.

---

## The Confusion Matrix

All classification metrics derive from the confusion matrix:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive']).plot()
```

```
                  Predicted Negative  Predicted Positive
Actual Negative        TN                  FP
Actual Positive        FN                  TP
```

- **TP** (True Positive): Correctly predicted positive
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Predicted positive, actually negative — Type I error
- **FN** (False Negative): Predicted negative, actually positive — Type II error

---

## Core Metrics

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```
Use when classes are balanced. Fails on imbalanced datasets.

---

### Precision (Positive Predictive Value)
```
Precision = TP / (TP + FP)
```
```python
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
```
"Of all predicted positives, how many are actually positive?"

High precision → few false alarms. Use when false positives are costly (spam filter: don't block real emails).

---

### Recall (Sensitivity / True Positive Rate)
```
Recall = TP / (TP + FN)
```
```python
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)
```
"Of all actual positives, how many did we find?"

High recall → few missed positives. Use when false negatives are costly (cancer detection: don't miss actual cases).

---

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
```python
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
```
Harmonic mean of precision and recall. Use when both false positives and false negatives matter equally. Good single metric for imbalanced datasets.

---

### F-Beta Score
```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```
```python
from sklearn.metrics import fbeta_score
fbeta_score(y_test, y_pred, beta=2)  # β=2 weighs recall 2× more
```
- β > 1: prioritize recall (medical diagnosis)
- β < 1: prioritize precision (spam detection)

---

## Classification Report

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Output:
```
              precision    recall  f1-score   support
           0       0.87      0.93      0.90       105
           1       0.84      0.73      0.78        54
    accuracy                           0.86       159
   macro avg       0.86      0.83      0.84       159
weighted avg       0.86      0.86      0.86       159
```

`weighted avg` weights each class by its support count — appropriate for imbalanced datasets.

---

## ROC Curve and AUC

The **ROC curve** shows the trade-off between True Positive Rate (recall) and False Positive Rate at different thresholds:

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
```

**AUC (Area Under Curve)**:
- AUC = 0.5: random classifier (diagonal line)
- AUC = 1.0: perfect classifier
- AUC > 0.8: good model

AUC is threshold-independent — it measures how well the model ranks positives above negatives overall.

---

## Precision-Recall Curve

For highly imbalanced datasets, Precision-Recall curves are more informative than ROC:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
```

---

## Choosing Thresholds

The default threshold is 0.5, but you can tune it:

```python
# Find threshold that maximizes F1
f1_scores = []
thresholds_range = np.linspace(0.1, 0.9, 100)

for t in thresholds_range:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t))

best_threshold = thresholds_range[np.argmax(f1_scores)]
print("Best threshold:", best_threshold)
```

---

## Metric Selection Guide

| Use Case | Primary Metric |
|----------|---------------|
| Balanced classes | Accuracy |
| Imbalanced classes | F1, ROC-AUC |
| False positives are costly (spam) | Precision |
| False negatives are costly (medical) | Recall |
| Comparing across thresholds | ROC-AUC |
| Severe imbalance (fraud, rare disease) | PR-AUC |

---

## Common Pitfalls

- **Never report only accuracy on imbalanced data** — it is meaningless.
- **AUC doesn't tell you absolute performance** — a model with AUC=0.85 may still have low precision at the operating threshold.
- **Micro vs. macro avg for multi-class**: macro treats all classes equally; micro weights by class frequency. Use macro for balanced, weighted for imbalanced.
