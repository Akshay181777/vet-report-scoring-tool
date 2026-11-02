# Veterinary Radiology Report Scoring Tool

## Overview

A Python-based project that uses Large Language Models (LLMs) to classify findings from veterinary thoracic radiology reports into predefined conditions.

The project focuses on generating structured Positive/Negative classifications and evaluating model predictions against manually prepared reference labels.

## Features

- LLM-based report classification using the Groq API.
- Structured JSON output for predefined conditions.
- Positive and Negative classification.
- JSON response validation and repair.
- Checkpoint-based processing to resume interrupted runs.
- Confusion matrix generation.
- Evaluation using:
  - True Positive (TP)
  - True Negative (TN)
  - False Positive (FP)
  - False Negative (FN)
  - Sensitivity
  - Specificity

## Technologies Used

- Python
- Pandas
- Groq API
- JSON
- OpenPyXL
- Scikit-learn concepts (confusion matrix evaluation)

## Project Structure

```text
vet-report-scoring-tool/
│
├── main.py
├── confusion_matrix.py
├── prompts.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Akshay181777/vet-report-scoring-tool.git
cd vet-report-scoring-tool
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Configure your Groq API key in a local `.env` file:

```text
GROQ_API_KEY=your_api_key
```

Do not commit the `.env` file.

## Execution

Run the classification script using your locally available input and reference-label workbooks:

```bash
python main.py input.xlsx ground_truth.xlsx --limit 100 -o output.xlsx
```

## Evaluation Method

The project compares LLM-generated classifications against manually prepared reference labels using CaseID matching.

A confusion matrix is generated for each predefined condition.

## Data Confidentiality

All original veterinary reports, datasets, manually prepared reference labels, generated predictions and checkpoint files are maintained locally and are not included in this repository.

## Status

Implementation includes LLM classification, checkpoint-based processing and confusion matrix generation. Evaluation is ongoing.
