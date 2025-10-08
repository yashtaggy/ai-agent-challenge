import pandas as pd
import pdfplumber
import re
import os

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses an ICICI bank statement PDF and extracts transaction data into a pandas DataFrame.

    Args:
        pdf_path (str): The path to the PDF bank statement file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted transactions with columns:
                      'Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'.
                      'Debit Amt' and 'Credit Amt' will have NaN for non-applicable transactions.
    """
    all_transactions = []

    CREDIT_KEYWORDS_CORE = ['SALARY', 'DEPOSIT', 'INTEREST', 'RECEIVED', 'TRANSFER FROM', 'CASH DEPOSIT', 'FUNDS IN']
    DEBIT_KEYWORDS_CORE = ['PAYMENT', 'PURCHASE', 'RECHARGE', 'BILL', 'TRANSFER TO', 'EMI', 'SERVICE CHARGE', 'SWIPE', 'WITHDRAWAL', 'UPI', 'OUT']

    transaction_line_pattern = re.compile(r'^(\d{2}-\d{2}-\d{4})\s+(.*?)\s+(\d+\.?\d*)\s+(\d+\.?\d*)$')

    # Initialize an empty DataFrame for early exit in case of errors
    empty_df = pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    empty_df['Date'] = pd.to_datetime(empty_df['Date']) # Ensure Date column has datetime dtype

    # Check if the file exists before attempting to open it.
    # This directly addresses the `[Errno 2] No such file or directory` error.
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'. Returning an empty DataFrame.")
        return empty_df

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: # Skip empty pages
                    continue

                lines = text.split('\n')

                in_transactions_section = False
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Identify the start of the transactions section by the header line.
                    if 'Date Description Debit Amt Credit Amt Balance' in line:
                        in_transactions_section = True
                        continue

                    if in_transactions_section:
                        match = transaction_line_pattern.match(line)
                        if match:
                            date_str, description, amount_str, balance_str = match.groups()
                            
                            description_upper = description.upper()
                            debit_amt = None
                            credit_amt = None
                            
                            is_credit_transaction = False
                            is_debit_transaction = False

                            # Rule 1: Specific phrases that unequivocally indicate a DEBIT
                            if 'CREDIT CARD PAYMENT' in description_upper:
                                is_debit_transaction = True
                            
                            # Rule 2: Check for core CREDIT keywords or "CREDIT" itself (if not already classified as debit)
                            if not is_debit_transaction:
                                if any(k in description_upper for k in CREDIT_KEYWORDS_CORE) or \
                                   ('CREDIT' in description_upper and 'CARD' not in description_upper):
                                    is_credit_transaction = True

                            # Rule 3: If not yet classified, check for core DEBIT keywords or "DEBIT" itself
                            if not is_credit_transaction and not is_debit_transaction:
                                if any(k in description_upper for k in DEBIT_KEYWORDS_CORE) or \
                                   ('DEBIT' in description_upper):
                                    is_debit_transaction = True
                            
                            # Assign the extracted amount to the appropriate column
                            if is_credit_transaction:
                                credit_amt = float(amount_str)
                            elif is_debit_transaction:
                                debit_amt = float(amount_str)
                            else:
                                # Fallback for unclassified transactions: assume debit.
                                debit_amt = float(amount_str)

                            all_transactions.append({
                                'Date': date_str,
                                'Description': description.strip(),
                                'Debit Amt': debit_amt,
                                'Credit Amt': credit_amt,
                                'Balance': float(balance_str)
                            })
    except pdfplumber.PDFSyntaxError:
        print(f"Error: The file at '{pdf_path}' is not a valid PDF or is corrupted. Returning an empty DataFrame.")
        return empty_df
    except Exception as e:
        print(f"An unexpected error occurred while processing '{pdf_path}': {e}. Returning an empty DataFrame.")
        return empty_df

    df = pd.DataFrame(all_transactions)

    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
        df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
        
        df = df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]
    else:
        # If no transactions were found, return an empty DataFrame with the correct schema
        df = empty_df

    return df