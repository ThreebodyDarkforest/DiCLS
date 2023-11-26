from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    f1_score,
    average_precision_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

SCALAR = ["accuracy", "precision", "recall", "f1_score", "avg_precision"]
CURVE = ["roc_curve", "micro_roc", "pr_curve"]

def evaluate(logits: np.ndarray, targets: np.ndarray, 
             num_classes: int, threshold: float=0.5,
             accuracy_on=True, precision_on=True, recall_on=True,
             pr_curve_on=False, f1_score_on=True, avg_precision_on=False,
             roc_curve_on=False):
    
    results = {"num_classes": num_classes}
    y_true_binary = label_binarize(targets, classes=list(range(targets.shape[-1])))

    y_pred_binary = (logits >= threshold).astype(int)
    accuracy = accuracy_score(targets, y_pred_binary)
    if accuracy_on:
        results.update({"accuracy": accuracy})

    precision = precision_score(y_true_binary, y_pred_binary, average='micro')
    if precision_on:
        results.update({"precision": precision})

    recall = recall_score(y_true_binary, y_pred_binary, average='micro')
    if recall_on:
        results.update({"recall": recall})

    f1 = f1_score(y_true_binary, y_pred_binary, average='micro')
    if f1_score_on:
        results.update({"f1_score": f1})

    average_precision = average_precision_score(y_true_binary, logits, average='micro')
    if avg_precision_on:
        results.update({"avg_precision": average_precision})
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr_micro, tpr_micro, _ = roc_curve(y_true_binary.ravel(), logits.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    if roc_curve_on:
        results.update({"roc_curve": (fpr, tpr, roc_auc)})
        results.update({"micro_roc": (fpr_micro, tpr_micro, roc_auc_micro)})
    
    if pr_curve_on:
        precision, recall = [], []
        thresholds = np.linspace(0, 1, 20)
        for thres in thresholds:
            pre = precision_score(targets, (logits >= thres).astype(int), average='micro')
            rec = recall_score(targets, (logits >= thres).astype(int), average='micro')
            precision.append(pre)
            recall.append(rec)
            #precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_binary)
        precision = np.vstack(precision)
        recall = np.vstack(recall)

        results.update({"pr_curve": (precision, recall, thresholds)})

    return results