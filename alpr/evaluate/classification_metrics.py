"""Métricas de clasificación (sin dependencias pesadas): exactitud, precisión y recall por clase,
matriz de confusión. Las usa el entrenamiento del clasificador de vehículos."""


def classification_report(y_true, y_pred, classes):
    """y_true / y_pred: listas de índices de clase. Devuelve un dict serializable."""
    n = len(classes)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred tienen distinto largo")
    matrix = [[0] * n for _ in range(n)]          # fila = clase real, columna = clase predicha
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(n))
    per_class = {}
    for i, name in enumerate(classes):
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(n))
        tp = matrix[i][i]
        per_class[name] = {
            "support": support,
            "recall": tp / support if support else None,          # None: no hay ejemplos de esa clase
            "precision": tp / predicted if predicted else None,    # None: nunca se predijo
        }
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {
        "n": total,
        "accuracy": correct / total if total else None,
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "per_class": per_class,
        "classes": list(classes),
        "confusion_matrix": matrix,
    }
