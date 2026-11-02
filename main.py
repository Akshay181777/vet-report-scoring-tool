
"""Groq report classification and Matrix 2 generation.

Run:
python main.py "CanineThorax.xlsx" "CanineThorax.xlsx" --limit 100 -o "CanineThorax_Groq_Matrix2.xlsx"

Successful case predictions are saved immediately to a JSONL checkpoint.
If the script stops, run the same command again; completed cases are skipped.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from json_repair import repair_json

from confusion_matrix import calculate_matrix, normalize_case_id


def find_col(df, names):
    """Find a dataframe column while ignoring spaces and underscores."""
    lookup = {
        str(column).lower().replace(" ", "").replace("_", ""): column
        for column in df.columns
    }

    for name in names:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in lookup:
            return lookup[key]

    return None


def parse_groq_json(content):
    """Parse valid JSON or attempt to repair malformed JSON from Groq."""
    if not content or not content.strip():
        raise ValueError("Groq returned an empty response.")

    content = content.strip()

    # Remove Markdown code fences, if the model included them.
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # First try standard JSON parsing.
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        print("The response was not valid JSON. Trying JSON repair...")
        repaired = repair_json(content)
        result = json.loads(repaired)

    if not isinstance(result, dict):
        raise ValueError("Groq response must be a JSON object.")

    return result


def classify_report(client, config, report):
    """Make one Groq request and validate all condition classifications."""
    conditions = config["conditions"]

    definitions = "\n".join(
        f"- {condition}: {definition}"
        for condition, definition in conditions.items()
    )

    system = "\n".join(config["global_instructions"])

    user = (
        f"Condition definitions:\n{definitions}\n\n"
        f"Radiology report:\n{report}\n\n"
        "Return one JSON object only. Include every condition name "
        "as a top-level key. Each value must be an object with "
        '"classification" set to exactly "Positive" or "Negative", '
        'and "reason" set to a short explanation. '
        "Do not use Markdown fences or text outside the JSON object."
    )

    # Exactly one API request per case. No automatic retries.
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=config.get("temperature", 0),
        max_completion_tokens=config.get("max_completion_tokens", 3000),
    )

    content = response.choices[0].message.content
    result = parse_groq_json(content)

    clean = {}

    for condition in conditions:
        if condition not in result:
            raise ValueError(
                f"Groq response is missing condition: {condition}"
            )

        item = result[condition]
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid response structure for condition: {condition}"
            )

        label = str(item.get("classification", "")).strip().title()
        if label not in {"Positive", "Negative"}:
            raise ValueError(
                f"Invalid classification for {condition}: {label!r}"
            )

        clean[condition] = {
            "classification": label,
            "reason": str(item.get("reason", "")).strip(),
        }

    return clean


def load_checkpoint(checkpoint_path):
    """Load completed predictions from the JSONL checkpoint file."""
    completed = {}

    if not checkpoint_path.exists():
        return completed

    with checkpoint_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                case_id = str(record["CaseID"])
                result = record["result"]

                if isinstance(result, dict):
                    completed[case_id] = result
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(
                    f"Warning: ignoring invalid checkpoint line "
                    f"{line_number}: {exc}"
                )

    return completed


def append_checkpoint(checkpoint_path, case_id, result):
    """Persist one successful case immediately."""
    record = {
        "CaseID": str(case_id),
        "result": result,
    }

    with checkpoint_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def build_output_rows(reports, conditions, completed):
    """Build prediction and detail rows from all completed checkpoint cases."""
    prediction_rows = []
    detail_rows = []

    for _, row in reports.iterrows():
        case_id = str(normalize_case_id(row["CaseID"]))

        if case_id not in completed:
            continue

        result = completed[case_id]
        wide_row = {"CaseID": row["CaseID"]}

        for condition in conditions:
            item = result[condition]
            classification = item["classification"]

            wide_row[condition] = classification
            detail_rows.append(
                {
                    "CaseID": row["CaseID"],
                    "Condition": condition,
                    "Classification": classification,
                    "Short reason": item.get("reason", ""),
                }
            )

        prediction_rows.append(wide_row)

    return pd.DataFrame(prediction_rows), pd.DataFrame(detail_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Groq predictions and Matrix 2."
    )

    parser.add_argument(
        "input_file",
        help="Workbook containing original radiologist reports",
    )
    parser.add_argument(
        "ground_truth_file",
        help="Workbook containing manual Gold Standard labels",
    )
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "-o",
        "--output",
        default="CanineThorax_Groq_Matrix2.xlsx",
    )

    args = parser.parse_args()

    for filename in (
        args.input_file,
        args.ground_truth_file,
        args.prompts,
    ):
        if not Path(filename).is_file():
            raise SystemExit(f"File not found: {filename}")

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise SystemExit("GROQ_API_KEY not found in .env")

    config = json.loads(
        Path(args.prompts).read_text(encoding="utf-8")
    )
    conditions = list(config["conditions"])

    reports = pd.read_excel(args.input_file, sheet_name=0)
    truth = pd.read_excel(args.ground_truth_file, sheet_name=0)

    report_case_col = find_col(reports, ["CaseID"])
    if report_case_col is None:
        raise SystemExit("Input workbook needs a CaseID column.")

    reports = (
        reports.rename(columns={report_case_col: "CaseID"})
        .dropna(subset=["CaseID"])
        .head(args.limit)
        .copy()
    )

    report_cols = {
        "Findings": find_col(
            reports,
            ["Findings (original radiologist report)", "Findings"],
        ),
        "Conclusions": find_col(
            reports,
            ["Conclusions (original radiologist report)", "Conclusions"],
        ),
        "Recommendations": find_col(
            reports,
            [
                "Recommendations (original radiologist report)",
                "Recommendations",
            ],
        ),
    }
    report_cols = {
        label: column
        for label, column in report_cols.items()
        if column is not None
    }

    if not report_cols:
        raise SystemExit(
            "No Findings/Conclusions/Recommendations report text columns found."
        )

    truth_case_col = find_col(truth, ["CaseID"])
    if truth_case_col is None:
        raise SystemExit("Gold Standard workbook needs a CaseID column.")

    truth = truth.rename(columns={truth_case_col: "CaseID"}).copy()

    # The checkpoint is stored beside the requested output workbook.
    output_path = Path(args.output)
    checkpoint_path = output_path.with_name(
        output_path.stem + "_checkpoint.jsonl"
    )

    completed = load_checkpoint(checkpoint_path)
    if completed:
        print(
            f"Loaded {len(completed)} completed cases from "
            f"{checkpoint_path.name}."
        )

    client = Groq(api_key=api_key)

    for index, (_, row) in enumerate(reports.iterrows(), start=1):
        original_case_id = row["CaseID"]
        case_key = str(normalize_case_id(original_case_id))

        if case_key in completed:
            print(
                f"Skipping CaseID {original_case_id} "
                f"({index}/{len(reports)}) - already saved."
            )
            continue

        sections = [
            f"{label}: {row[column]}"
            for label, column in report_cols.items()
            if pd.notna(row[column]) and str(row[column]).strip()
        ]
        report = "\n\n".join(sections)

        if not report:
            raise SystemExit(
                f"CaseID {original_case_id} has no report text."
            )

        print(
            f"Calling Groq for CaseID {original_case_id} "
            f"({index}/{len(reports)})"
        )

        try:
            result = classify_report(client, config, report)
        except Exception as exc:
            print(
                f"\nStopped at CaseID {original_case_id}. "
                "No automatic retry was made."
            )
            print(f"Reason: {exc}")
            print(
                f"Previously completed cases are saved in: "
                f"{checkpoint_path.resolve()}"
            )
            print(
                "Run the same command again after resolving the issue; "
                "saved cases will be skipped."
            )
            raise SystemExit(1) from exc

        # Save immediately after successful classification.
        append_checkpoint(checkpoint_path, case_key, result)
        completed[case_key] = result

        print(
            f"Saved CaseID {original_case_id} "
            f"({len(completed)} completed in checkpoint)."
        )

    # Build the final workbook using all completed predictions.
    predictions, details = build_output_rows(
        reports, conditions, completed
    )

    if predictions.empty:
        raise SystemExit("No completed predictions are available.")

    predictions["CaseID"] = normalize_case_id(predictions["CaseID"])
    truth["CaseID"] = normalize_case_id(truth["CaseID"])

    selected_ids = set(predictions["CaseID"])
    truth_selected = truth[truth["CaseID"].isin(selected_ids)].copy()

    matrix, matched, missing, extra = calculate_matrix(
        predictions,
        truth_selected,
        conditions,
    )

    qa = pd.DataFrame(
        [
            ["Reports classified", len(predictions)],
            ["Conditions", len(conditions)],
            [
                "Expected prediction labels",
                len(predictions) * len(conditions),
            ],
            ["Actual prediction labels", len(details)],
            ["CaseIDs matched", matched],
            [
                "Gold Standard IDs missing predictions",
                ", ".join(missing) or "None",
            ],
            ["Extra prediction IDs", ", ".join(extra) or "None"],
            ["Groq model", config["model"]],
            ["Checkpoint file", checkpoint_path.name],
        ],
        columns=["QA check", "Result"],
    )

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        predictions.to_excel(
            writer, sheet_name="Sheet1", index=False
        )
        details.to_excel(
            writer, sheet_name="Detailed Predictions", index=False
        )
        matrix.to_excel(
            writer, sheet_name="Matrix 2", index=False
        )
        qa.to_excel(
            writer, sheet_name="QA", index=False
        )

        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for cells in worksheet.columns:
                width = max(
                    len(str(cell.value or "")) for cell in cells
                ) + 2
                worksheet.column_dimensions[
                    cells[0].column_letter
                ].width = min(max(width, 12), 55)

        matrix_ws = writer.sheets["Matrix 2"]
        for column_name in ("Sensitivity", "Specificity"):
            if column_name in matrix.columns:
                column_index = matrix.columns.get_loc(column_name) + 1
                for row_index in range(2, len(matrix) + 2):
                    matrix_ws.cell(
                        row_index, column_index
                    ).number_format = "0.00%"

    print("\nFinished.")
    print("Output:", output_path.resolve())
    print("Matrix 2 matched cases:", matched)
    print("Checkpoint:", checkpoint_path.resolve())


if __name__ == "__main__":
    main()
