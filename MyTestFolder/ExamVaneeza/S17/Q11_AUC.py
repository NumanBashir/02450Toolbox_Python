def calculate_auc(thresholds, tpr, fpr):
    """
    Beregn AUC for en ROC-kurve ved hjælp af trapezmetoden.

    Parametre:
    thresholds (list): Liste af thresholds.
    tpr (list): Liste af True Positive Rate værdier.
    fpr (list): Liste af False Positive Rate værdier.

    Returner:
    float: Area Under Curve (AUC).
    """
    if len(thresholds) != len(tpr) or len(thresholds) != len(fpr):
        raise ValueError("List lengths of thresholds, tpr, and fpr must be the same.")

    auc = 0.0
    for i in range(1, len(thresholds)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return auc

# Eksempel på brug:
thresholds = [6, 5, 4, 3] #kan ignoreres
tpr = [0, 3/15, 13/15, 1] #ændre 
fpr = [0, 2/17, 4/17, 1]# ændre

auc = calculate_auc(thresholds, tpr, fpr)
print(f"AUC: {auc:.3f}")
