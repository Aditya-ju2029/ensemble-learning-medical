import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def load_predictions(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        return np.array([[float(v) for v in row[3:]] for row in reader])


def load_labels(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        return {row[0]: row[1] for row in reader}


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def inverted_bell(x, a, b, c):
    return (1 / a) * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def objective(params, preds, y):
    a, b, c = params
    w = inverted_bell(preds, a, b, c)
    ens = np.average(preds, axis=0, weights=w)
    return -accuracy_score(y, np.argmax(ens, axis=1))


labels = load_labels("true_labels_with_names.csv")
names = list(labels.keys())

preds = np.array([
    softmax(load_predictions("predictions_inception.csv")),
    softmax(load_predictions("predictions_vgg16.csv")),
    softmax(load_predictions("predictions_resnet50.csv"))
])

class_map = {"Cyst": 0, "Normal": 1, "Stone": 2, "Tumor": 3}
y = np.array([class_map[labels[n]] for n in names])
class_names = list(class_map.keys())

kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics, cms = [], {}

for fold, (tr, va) in enumerate(kf.split(preds[0]), 1):

    res = minimize(
        objective,
        x0=[1, 0.5, 0.5],
        args=(preds[:, tr], y[tr]),
        bounds=[(0.1, 10), (0, 1), (0.1, 1)]
    )

    a, b, c = res.x
    w = inverted_bell(preds[:, va], a, b, c)
    ens = np.average(preds[:, va], axis=0, weights=w)
    y_hat = np.argmax(ens, axis=1)

    acc = accuracy_score(y[va], y_hat)
    prec = precision_score(y[va], y_hat, average="macro", zero_division=0)
    rec = recall_score(y[va], y_hat, average="macro", zero_division=0)
    f1 = f1_score(y[va], y_hat, average="macro", zero_division=0)

    metrics.append([fold, acc, prec, rec, f1])

    report = classification_report(
        y[va], y_hat, target_names=class_names,
        output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    df.loc["Accuracy", ["precision", "recall", "f1-score"]] = acc
    df.loc["Accuracy", "support"] = len(va)
    df.to_csv(f"classwise_metrics_fold_{fold}.csv")

    cms[f"Fold_{fold}"] = confusion_matrix(y[va], y_hat)

pd.DataFrame(
    metrics,
    columns=["Fold", "Accuracy", "Precision", "Recall", "F1-score"]
).to_csv("ensemble_fold_metrics.csv", index=False)

for k, cm in cms.items():
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Ensemble Confusion Matrix ({k})")
    plt.tight_layout()
    plt.show()
