
"""Matrix 2 calculations."""

import re
import pandas as pd


def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_case_id(value):
    """Normalize either a single CaseID or a Pandas Series."""

    if isinstance(value, pd.Series):
        return (
            value.astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    if pd.isna(value):
        return ""

    return re.sub(r"\.0$", "", str(value).strip())


def normalize_label(value):
    """Convert Positive/Negative labels into binary values."""

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"positive", "abnormal", "yes", "y", "1", "true"}:
        return 1

    if value in {"negative", "normal", "no", "n", "0", "false"}:
        return 0

    return None


def calculate_matrix(predictions, truth, conditions):
    """Compare LLM predictions against manual Gold Standard labels."""

    predictions = predictions.copy()
    truth = truth.copy()

    predictions["CaseID"] = normalize_case_id(predictions["CaseID"])
    truth["CaseID"] = normalize_case_id(truth["CaseID"])

    if predictions["CaseID"].duplicated().any():
        raise ValueError("Duplicate CaseIDs found in predictions.")

    if truth["CaseID"].duplicated().any():
        raise ValueError("Duplicate CaseIDs found in Gold Standard.")

    merged = predictions.merge(
        truth,
        on="CaseID",
        suffixes=("_pred", "_truth"),
        how="inner",
        validate="one_to_one"
    )

    missing = sorted(set(truth["CaseID"]) - set(predictions["CaseID"]))
    extra = sorted(set(predictions["CaseID"]) - set(truth["CaseID"]))

    rows = []

    for condition in conditions:

        if condition not in predictions.columns:
            raise ValueError(
                f"Missing prediction column: {condition}"
            )

        if condition not in truth.columns:
            raise ValueError(
                f"Missing Gold Standard column: {condition}"
            )

        p = merged[f"{condition}_pred"].map(normalize_label)
        t = merged[f"{condition}_truth"].map(normalize_label)

        truth_valid = t.notna()

        pos_gt = int((t[truth_valid] == 1).sum())
        neg_gt = int((t[truth_valid] == 0).sum())

        valid = p.notna() & t.notna()

        p = p[valid].astype(int)
        t = t[valid].astype(int)

        tp = int(((p == 1) & (t == 1)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        tn = int(((p == 0) & (t == 0)).sum())
        fp = int(((p == 1) & (t == 0)).sum())

        rows.append({
            "condition": condition,
            "True Positive": tp,
            "False Negative": fn,
            "True Negative": tn,
            "False Positive": fp,
            "Sensitivity": tp / (tp + fn) if tp + fn else None,
            "Specificity": tn / (tn + fp) if tn + fp else None,
            "Check": tp + fn + tn + fp,
            "Positive Ground Truth": pos_gt,
            "Negative Ground Truth": neg_gt,
            "Ground Truth Check": pos_gt + neg_gt
        })

    return pd.DataFrame(rows), len(merged), missing, extra
