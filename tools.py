import pandas as pd
import importlib.util
import os
import shutil
import subprocess
from typing import Dict, Any

# --- Paths (relative to the project root) ---
PARSERS_DIR = "custom_parsers"
DATA_DIR = "data"

def save_code(code: str, target_file: str) -> Dict[str, Any]:
    """Saves the generated Python code to the target file."""
    try:
        if not os.path.exists(PARSERS_DIR):
            os.makedirs(PARSERS_DIR)
        
        # Ensure the file is a Python file
        full_path = os.path.join(PARSERS_DIR, target_file)
        
        with open(full_path, "w") as f:
            f.write(code)
            
        return {
            "status": "success",
            "message": f"Code successfully written to {full_path}"
        }
    except Exception as e:
        return {
            "status": "failure",
            "message": f"Error saving code: {e}"
        }

def run_test(bank_name: str, max_attempts: int, current_attempt: int) -> Dict[str, Any]:
    """
    Imports the generated parser, runs the 'parse' function, and compares
    the output with the ground truth CSV.
    """
    parser_file = f"{bank_name}_parser.py"
    module_name = bank_name + "_parser"
    parser_path = os.path.join(PARSERS_DIR, parser_file)
    pdf_path = os.path.join(DATA_DIR, bank_name, f"{bank_name}_sample.pdf")
    csv_path = os.path.join(DATA_DIR, bank_name, f"{bank_name}_sample.csv")

    print(f"\n--- Running Test (Attempt {current_attempt}/{max_attempts}) ---")

    # 1. Check if the parser file exists
    if not os.path.exists(parser_path):
        return {
            "test_result": "FAIL",
            "reason": f"Parser file not found at {parser_path}. Agent must first generate code.",
            "feedback": "Please generate the Python code for the parser file first."
        }

    # 2. Dynamic Import
    try:
        spec = importlib.util.spec_from_file_location(module_name, parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        parse_func = getattr(parser_module, "parse")
    except Exception as e:
        return {
            "test_result": "FAIL",
            "reason": f"Failed to import parser or find parse() function.",
            "feedback": f"Import/Execution Error: {e}\nCheck for syntax errors, missing imports (like 'pandas' as 'pd'), or if 'parse(pdf_path)' is missing."
        }

    # 3. Execution and Comparison
    try:
        # Load expected DataFrame first
        expected_df = pd.read_csv(csv_path)

        # Run the agent-generated parser
        actual_df = parse_func(pdf_path)

        # T4: Test Assert parse() output equals the provided CSV via DataFrame.equals
        if actual_df.equals(expected_df):
            return {
                "test_result": "PASS",
                "reason": "The generated DataFrame exactly matches the expected CSV.",
                "feedback": f"SUCCESS! Parser for {bank_name} is complete."
            }
        else:
            # Generate detailed feedback for the agent
            
            # Check column mismatch
            if not actual_df.columns.equals(expected_df.columns):
                 col_feedback = (
                    f"Column Mismatch:\n"
                    f"Expected Columns: {list(expected_df.columns)}\n"
                    f"Actual Columns: {list(actual_df.columns)}"
                )
            else:
                col_feedback = "Columns match, but data content differs."
                
            # Compare first few rows
            actual_head = actual_df.head().to_markdown(index=False)
            expected_head = expected_df.head().to_markdown(index=False)
            
            feedback = (
                f"{col_feedback}\n\n"
                f"--- Expected DataFrame Head ---\n{expected_head}\n\n"
                f"--- Actual DataFrame Head ---\n{actual_head}\n\n"
                f"Review your logic for extracting data from the PDF."
            )
            
            return {
                "test_result": "FAIL",
                "reason": "DataFrame content mismatch detected.",
                "feedback": feedback
            }

    except Exception as e:
        # This catches errors during parsing (e.g., library errors, index errors)
        return {
            "test_result": "FAIL",
            "reason": f"Runtime error during parser execution.",
            "feedback": f"Execution Traceback/Error: {e}. Analyze the error and correct your parsing code."
        }

# Use this to clear the custom parser for fresh runs
def cleanup(bank_name: str) -> None:
    """Removes the generated parser file."""
    parser_path = os.path.join(PARSERS_DIR, f"{bank_name}_parser.py")
    if os.path.exists(parser_path):
        os.remove(parser_path)
        print(f"Cleaned up {parser_path}")