
"""Generate a confusion matrix for 100 selected cases.

Predictions come from Sheet1 of the original workbook.
Ground-truth labels come from the supplied ground-truth workbook.
All calculations run locally; no AI API is called.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_name(name):
    """Ignore capitalization, spaces, and punctuation in column names."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalize_case_id(series):
    """Standardize CaseIDs so IDs match between Excel files."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_label(value):
    """Convert positive/abnormal labels to 1 and negative/normal to 0."""
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    if label in {"positive", "abnormal", "yes", "y", "1", "true"}:
        return 1

    if label in {"negative", "normal", "no", "n", "0", "false"}:
        return 0

    return None


def is_metadata_column(column):
    """Identify columns that are not condition labels."""
    metadata = {
        "caseid",
        "findingsoriginalradiologistreport",
        "conclusionsoriginalradiologistreport",
        "recommendationsoriginalradiologistreport",
    }
    return normalize_name(column) in metadata


def create_confusion_matrix(prediction_file, ground_truth_file):
    """Calculate metrics for every ground-truth condition."""
    # Sheet1 has the predictions for all 21 conditions.
    predictions = pd.read_excel(
        prediction_file,
        sheet_name="Sheet1",
    )

    # The supplied ground-truth workbook contains the selected 100 cases.
    ground_truth = pd.read_excel(
        ground_truth_file,
        sheet_name=0,
    )

    # Ensure both workbooks have a CaseID column.
    if "CaseID" not in predictions.columns:
        raise ValueError("Sheet1 has no CaseID column.")

    if "CaseID" not in ground_truth.columns:
        raise ValueError("The ground-truth sheet has no CaseID column.")

    predictions = predictions.dropna(subset=["CaseID"]).copy()
    ground_truth = ground_truth.dropna(subset=["CaseID"]).copy()

    predictions["CaseID"] = normalize_case_id(predictions["CaseID"])
    ground_truth["CaseID"] = normalize_case_id(ground_truth["CaseID"])

    # The ground-truth workbook defines the cases to evaluate.
    selected_ids = set(ground_truth["CaseID"])
    predictions = predictions[
        predictions["CaseID"].isin(selected_ids)
    ].copy()

    # Check duplicate IDs only among the selected cases.
    if ground_truth["CaseID"].duplicated().any():
        raise ValueError(
            "The ground-truth workbook contains duplicate CaseIDs."
        )

    if predictions["CaseID"].duplicated().any():
        raise ValueError(
            "Duplicate prediction CaseIDs exist among the selected cases."
        )

    prediction_ids = set(predictions["CaseID"])
    unmatched_ids = sorted(selected_ids - prediction_ids)

    # Find prediction columns by normalized condition name.
    prediction_lookup = {
        normalize_name(column): column
        for column in predictions.columns
        if not is_metadata_column(column)
    }

    # Keep every condition in ground truth, even if a prediction
    # column cannot be found.
    condition_pairs = []
    for truth_column in ground_truth.columns:
        if is_metadata_column(truth_column):
            continue

        prediction_column = prediction_lookup.get(
            normalize_name(truth_column)
        )
        condition_pairs.append((prediction_column, truth_column))

    # Match cases using CaseID.
    # Only include columns needed for the confusion matrix.
    prediction_data = predictions[["CaseID"]].copy()
    truth_data = ground_truth[["CaseID"]].copy()

    for index, (prediction_column, truth_column) in enumerate(
        condition_pairs
    ):
        if prediction_column is None:
            # Preserve the condition, but don't invent predictions.
            prediction_data[f"pred_{index}"] = None
        else:
            prediction_data[f"pred_{index}"] = predictions[
                prediction_column
            ].map(normalize_label)

        truth_data[f"truth_{index}"] = ground_truth[
            truth_column
        ].map(normalize_label)

    matched = prediction_data.merge(
        truth_data,
        on="CaseID",
        how="inner",
        validate="one_to_one",
    )

    results = []

    for index, (_, truth_column) in enumerate(condition_pairs):
        prediction_values = matched[f"pred_{index}"]
        truth_values = matched[f"truth_{index}"]

        # Ground-truth totals are counted independently of predictions.
        valid_truth = truth_values.notna()
        positive_truth = int((truth_values[valid_truth] == 1).sum())
        negative_truth = int((truth_values[valid_truth] == 0).sum())

        # Confusion-matrix counts require both labels to be available.
        valid_pairs = prediction_values.notna() & truth_values.notna()
        pred = prediction_values[valid_pairs].astype(int)
        actual = truth_values[valid_pairs].astype(int)

        tp = int(((pred == 1) & (actual == 1)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())

        check = tp + fn + tn + fp

        sensitivity = (
            tp / (tp + fn) if (tp + fn) > 0 else None
        )
        specificity = (
            tn / (tn + fp) if (tn + fp) > 0 else None
        )

        results.append({
            "condition": truth_column,
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Check": check,
            "Positive Ground Truth": positive_truth,
            "Negative Ground Truth": negative_truth,
            "Ground Truth Check": positive_truth + negative_truth,
        })

    return pd.DataFrame(results), len(matched), unmatched_ids


def main():
    """Read file paths from the command line and save the output."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a confusion matrix for selected ground-truth cases."
        )
    )
    parser.add_argument(
        "prediction_file",
        help="Original Excel workbook containing Sheet1 predictions",
    )
    parser.add_argument(
        "ground_truth_file",
        help="Excel workbook containing selected ground-truth cases",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="my_confusion_matrix_100.xlsx",
        help="Output Excel filename",
    )

    args = parser.parse_args()

    prediction_path = Path(args.prediction_file)
    truth_path = Path(args.ground_truth_file)
    output_path = Path(args.output)

    if not prediction_path.is_file():
        raise SystemExit(f"Prediction workbook not found: {prediction_path}")

    if not truth_path.is_file():
        raise SystemExit(f"Ground-truth workbook not found: {truth_path}")

    # All calculations are local. No API calls or API credits are used.
    try:
        result, matched_count, unmatched_ids = create_confusion_matrix(
            prediction_path,
            truth_path,
        )
    except Exception as error:
        raise SystemExit(f"Could not create confusion matrix: {error}")

    # Save every condition in the output workbook.
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(
            writer,
            sheet_name="Confusion Matrix",
            index=False,
        )

        worksheet = writer.sheets["Confusion Matrix"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        # Display sensitivity and specificity as percentages.
        for column_name in ["Sensitivity", "Specificity"]:
            column_index = result.columns.get_loc(column_name) + 1

            for row_index in range(2, len(result) + 2):
                worksheet.cell(
                    row=row_index,
                    column=column_index,
                ).number_format = "0.00%"

        worksheet.column_dimensions["A"].width = 36

    print(f"Ground-truth cases: {len(pd.read_excel(truth_path))}")
    print(f"Cases matched: {matched_count}")
    print(f"Conditions included: {len(result)}")
    print(f"Output saved to: {output_path.resolve()}")

    if unmatched_ids:
        print(f"CaseIDs without predictions: {unmatched_ids}")


if __name__ == "__main__":
    main()
