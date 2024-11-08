import pandas as pd
import re
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def load_mapping_rules(csv_file):
    """
    Load mapping rules from a CSV file into a dictionary.
    Rules are sorted by Priority (lower number = higher Priority).
   
    Args:
        csv_file (str): Path to the CSV file containing mapping rules.
       
    Returns:
        tuple: (rules_dict, original_df) - Dictionary of rules and original DataFrame for updating
    """
    try:
        rules_df = pd.read_csv(csv_file)
        
        # Add statistics columns if they don't exist
        if 'CurrentRunMatches' not in rules_df.columns:
            rules_df['CurrentRunMatches'] = 0
        if 'TotalMatches' not in rules_df.columns:
            rules_df['TotalMatches'] = 0
            
        # Reset CurrentRunMatches for new run
        rules_df['CurrentRunMatches'] = 0
        
        # Sort by Priority
        rules_df = rules_df.sort_values('Priority', ascending=True)
        
        # Create rules dictionary
        rules_dict = {row['Pattern']: {
            'values': row.drop(['Pattern', 'Priority', 'CurrentRunMatches', 'TotalMatches']).to_dict(),
            'index': idx
        } for idx, row in rules_df.iterrows()}
        
        return rules_dict, rules_df
        
    except Exception as e:
        logging.error(f"Error loading mapping rules: {e}")
        return {}, None

def update_mapping_statistics(rules_df, pattern_matches):
    """
    Update the statistics in the mapping rules DataFrame.
    
    Args:
        rules_df (pd.DataFrame): The original mapping rules DataFrame
        pattern_matches (dict): Dictionary counting pattern matches
    
    Returns:
        pd.DataFrame: Updated DataFrame
    """
    # Update CurrentRunMatches
    for pattern, count in pattern_matches.items():
        mask = rules_df['Pattern'] == pattern
        rules_df.loc[mask, 'CurrentRunMatches'] = count
        rules_df.loc[mask, 'TotalMatches'] += count
    
    return rules_df

def save_mapping_rules(rules_df, csv_file):
    """
    Save updated mapping rules back to CSV file.
    
    Args:
        rules_df (pd.DataFrame): The updated mapping rules DataFrame
        csv_file (str): Path to the CSV file
    """
    try:
        rules_df.to_csv(csv_file, index=False)
        logging.info(f"Updated mapping rules saved to {csv_file}")
    except Exception as e:
        logging.error(f"Error saving mapping rules: {e}")

def clean_description(description):
    """
    Remove date (DDMMM) and time (HH:MM) patterns from description.
    
    Args:
        description (str): The description text to clean
        
    Returns:
        str: Cleaned description with date and time patterns removed
    """
    if not isinstance(description, str):
        return description
        
    # Pattern for DDMMM (e.g., 22JAN)
    date_pattern = r'\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
    
    # Pattern for HH:MM (e.g., 15:30)
    time_pattern = r'\d{2}:\d{2}'
    
    # Remove date pattern (case insensitive)
    cleaned = re.sub(date_pattern, '', description, flags=re.IGNORECASE)
    
    # Remove time pattern
    cleaned = re.sub(time_pattern, '', cleaned)
    
    # Remove any extra whitespace created by the removals
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def apply_mapping(row, mapping_rules, pattern_matches):
    """
    Apply mapping rules to a single row of transaction data.
    First cleans the Description field by removing date/time patterns,
    then applies mapping rules if they exist.
   
    Args:
        row (pd.Series): A row of transaction data.
        mapping_rules (dict): A dictionary of mapping rules.
        pattern_matches (dict): Dictionary to track pattern matches
   
    Returns:
        pd.Series: The updated row after applying mapping rules.
    """
    # First, clean the description (this happens regardless of mapping)
    original_description = row['Description']
    cleaned_description = clean_description(original_description)
    
    # Update the description field with the cleaned version
    row['Description'] = cleaned_description
    
    # If the description was already mapped, return
    if row.get('changed', False):
        return row

    # Create combined text with cleaned description for pattern matching
    combined_text = f"{row['Category']} {row['SubCategory']} {cleaned_description}"
    
    # Apply mapping rules
    for pattern, rule_info in mapping_rules.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            for key, value in rule_info['values'].items():
                if pd.notna(value):
                    if key in row.index and key in ['Category', 'SubCategory', 'Description']:
                        row[key] = value
            row['changed'] = True
            
            # Increment pattern match counter
            pattern_matches[pattern] = pattern_matches.get(pattern, 0) + 1
            
            logging.debug(f"Applied rule with pattern '{pattern}' to transaction: {cleaned_description} -> {rule_info['values'].get('Description', cleaned_description)}")
            break
            
    # Log significant description changes (not related to date/time removal)
    if original_description != cleaned_description:
        logging.debug(f"Cleaned description: '{original_description}' -> '{cleaned_description}'")
        
    return row

def extract_bank_owner(filename):
    """
    Extract bank and owner information from filename.
    
    Args:
        filename (str): The input filename
        
    Returns:
        tuple: (bank, owner) - extracted and truncated to 20 characters each
    """
    # Remove file extension and split by space
    name_parts = os.path.splitext(filename)[0].split(' ', 1)
    
    # Extract bank and owner, ensuring they don't exceed 20 characters
    bank = (name_parts[0][:20] if name_parts else "").strip()
    owner = (name_parts[1][:20] if len(name_parts) > 1 else "").strip()
    
    return bank, owner

def process_bank_statement(input_file, output_file, mapping_rules, pattern_matches):
    """
    Process a bank statement CSV file and apply mapping rules to categorize transactions.
   
    Args:
        input_file (str): Path to the input CSV file containing bank transactions.
        output_file (str): Path to the output CSV file to save processed transactions.
        mapping_rules (dict): A dictionary of mapping rules.
        pattern_matches (dict): Dictionary to track pattern matches
        
    Returns:
        tuple: (total_records, changed_records) - Statistics about the processing
    """
    try:
        # Read just the first row to get column names
        headers = pd.read_csv(input_file, nrows=0).columns.tolist()
        
        # Define the required columns we want to keep (first 7 fields)
        required_columns = ['Category', 'SubCategory', 'Description', 'Debit', 'Credit', 'Balance', 'Date']
        
        # Check if there are extra fields
        extra_fields = len(headers) > 7
        if extra_fields:
            logging.info(f"Found {len(headers)} fields in {input_file}. Keeping only first 7 fields.")
            
        # Read the CSV with only the required columns
        df = pd.read_csv(input_file, usecols=required_columns)
        logging.info(f"Successfully read input file: {input_file}")

        # Initialize changed column - set to False even if fields were dropped
        df['changed'] = False

        # Combine Debit and Credit into a single Amount column
        df['Amount'] = df['Debit'].fillna(0) - df['Credit'].fillna(0)
        
        # Extract bank and owner from filename
        bank, owner = extract_bank_owner(os.path.basename(input_file))
        
        # Add Bank and Owner columns
        df['Bank'] = bank
        df['Owner'] = owner
        
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return 0, 0

    df = df.apply(lambda row: apply_mapping(row, mapping_rules, pattern_matches), axis=1)

    # Calculate statistics
    total_records = len(df)
    changed_records = df['changed'].sum()

    try:
        # Ensure columns are in the desired order
        columns_order = ['Category', 'SubCategory', 'Description', 'Amount', 'Date', 'Bank', 'Owner']
        if 'changed' in df.columns:
            df = df[columns_order + ['changed']]
        else:
            df = df[columns_order]
            
        df.to_csv(output_file, index=False)
        logging.info(f"Processed data saved to {output_file}")
        return total_records, changed_records
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        return 0, 0

def process_directory(input_dir, output_dir, mapping_rules_file):
    """
    Process all CSV files in the input directory and save results to the output directory.
    
    Args:
        input_dir (str): Path to the directory containing input CSV files
        output_dir (str): Path to the directory where processed files will be saved
        mapping_rules_file (str): Path to the CSV file containing mapping rules
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load mapping rules
    mapping_rules, rules_df = load_mapping_rules(mapping_rules_file)
    if not mapping_rules:
        logging.error("No mapping rules loaded. Exiting.")
        return
    
    # Dictionary to track pattern matches across all files
    pattern_matches = {}
    
    # Initialize counters for overall statistics
    total_files = 0
    total_records_processed = 0
    total_records_changed = 0
    
    # Process each CSV file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"Processed_{filename}")
            
            logging.info(f"\nProcessing file: {filename}")
            total_records, changed_records = process_bank_statement(
                input_path, output_path, mapping_rules, pattern_matches
            )
            
            # Update statistics
            total_files += 1
            total_records_processed += total_records
            total_records_changed += changed_records
            
            # Log individual file statistics
            logging.info(f"File statistics for {filename}:")
            logging.info(f"  Total records: {total_records}")
            logging.info(f"  Records modified: {changed_records}")
            logging.info(f"  Records unchanged: {total_records - changed_records}")
    
    # Update and save mapping rules with new statistics
    if rules_df is not None:
        updated_rules_df = update_mapping_statistics(rules_df, pattern_matches)
        save_mapping_rules(updated_rules_df, mapping_rules_file)
    
    # Log overall statistics
    logging.info("\nOverall Processing Statistics:")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Total records processed: {total_records_processed}")
    logging.info(f"Total records modified: {total_records_changed}")
    logging.info(f"Total records unchanged: {total_records_processed - total_records_changed}")
    
    # Log pattern match statistics
    logging.info("\nPattern Match Statistics:")
    for pattern, count in pattern_matches.items():
        logging.info(f"Pattern '{pattern}': {count} matches")

# Example usage
input_directory = "bank_statements"
output_directory = "processed_statements"
mapping_rules_file = "mapping-rules.csv"

process_directory(input_directory, output_directory, mapping_rules_file)