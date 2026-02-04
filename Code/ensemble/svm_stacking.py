import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

data = pd.read_csv("probabilities_for_true_class.csv")

X = data.drop(columns=["Filename", "True Class"])
y = data["True Class"]
names = data["Filename"]
class_names = sorted(y.unique())

kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics, cms = [], {}

for fold, (tr, va) in enumerate(kf.split(X), 1):

    model = SVC(kernel="linear", class_weight="balanced", random_state=42)
    model.fit(X.iloc[tr], y.iloc[tr])
    y_hat = model.predict(X.iloc[va])

    acc = accuracy_score(y.iloc[va], y_hat)
    prec = precision_score(y.iloc[va], y_hat, average="macro", zero_division=0)
    rec = recall_score(y.iloc[va], y_hat, average="macro", zero_division=0)
    f1 = f1_score(y.iloc[va], y_hat, average="macro", zero_division=0)

    metrics.append([fold, acc, prec, rec, f1])

    report = classification_report(
        y.iloc[va], y_hat, target_names=class_names,
        output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    df.loc["Accuracy", ["precision", "recall", "f1-score"]] = acc
    df.loc["Accuracy", "support"] = len(va)
    df.to_csv(f"svm_classwise_metrics_fold_{fold}.csv")

    cms[f"Fold_{fold}"] = confusion_matrix(
        y.iloc[va], y_hat, labels=class_names
    )

pd.DataFrame(
    metrics,
    columns=["Fold", "Accuracy", "Precision", "Recall", "F1-score"]
).to_csv("svm_fold_metrics.csv", index=False)

for k, cm in cms.items():
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SVM Confusion Matrix ({k})")
    plt.tight_layout()
    plt.show()
