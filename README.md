
# Vetology Report Scoring Tool

## Overview
A Python command-line tool that compares predicted veterinary
findings with ground-truth labels and generates a confusion matrix.

## Features
- Reads Excel prediction and ground-truth workbooks.
- Evaluates selected cases by CaseID.
- Calculates TP, FN, TN, FP, sensitivity, and specificity.
- Exports results to an Excel workbook.
- Performs calculations locally without sending veterinary data
  to an AI service.

## Requirements
- Python 3.13
- pandas
- openpyxl
- flake8

## Installation
Create and activate a virtual environment, then run:

    python -m pip install -r requirements.txt

## Usage
Run the program with the paths to the prediction and ground-truth files:

    python main.py "path/to/predictions.xlsx" "path/to/FelineGroundTruth.xlsx" -o "my_confusion_matrix_100.xlsx"

## Confidentiality
Do not upload Vetology data, Excel workbooks, API keys, or
environment files to GitHub. Keep input and output data local.
