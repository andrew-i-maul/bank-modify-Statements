import pandas as pd
import re
import logging
import os
from pathlib import Path
from datetime import datetime

[... previous logging and mapping functions remain the same ...]

def merge_output_files(output_dir):
    """
    Merge all processed CSV files in the output directory into a single file.
    Only keeps headers from the first file.
    
    Args:
        output_dir (str): Directory containing the processed CSV files
        
    Returns:
        str: Path to the merged file
    """
    # Create timestamp for merged filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_file = os.path.join(output_dir, f'Merged_Transactions_{timestamp}.csv')
    
    try:
        # Get list of all processed CSV files
        csv_files = [f for f in os.listdir(output_dir) 
                    if f.startswith('Processed_') and f.endswith('.csv')]
        
        if not csv_files:
            logging.warning("No processed files found to merge")
            return None
            
        logging.info(f"Found {len(csv_files)} files to merge")
        
        # Read and merge all CSV files
        all_data = []
        for i, filename in enumerate(csv_files):
            file_path = os.path.join(output_dir, filename)
            # First file includes headers, others don't
            header = 0 if i == 0 else None
            df = pd.read_csv(file_path, header=header)
            all_data.append(df)
            logging.info(f"Added {filename} to merge ({len(df)} rows)")
            
        # Concatenate all DataFrames
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Save merged file
        merged_df.to_csv(merged_file, index=False)
        logging.info(f"Created merged file: {merged_file}")
        logging.info(f"Total rows in merged file: {len(merged_df)}")
        
        return merged_file
        
    except Exception as e:
        logging.error(f"Error merging files: {e}")
        return None

def process_directory(input_dir, output_dir, mapping_rules_file):
    """
    Process all CSV files in the input directory and save results to the output directory.
    Also creates a merged file of all processed transactions.
    
    Args:
        input_dir (str): Path to the directory containing input CSV files
        output_dir (str): Path to the directory where processed files will be saved
        mapping_rules_file (str): Path to the CSV file containing mapping rules
    """
    # Set up logging first
    log_file = setup_logging(output_dir)
    
    # Log start of processing with configuration details
    logging.info("Starting bank statement processing")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Mapping rules file: {mapping_rules_file}")
    
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
    
    # Create merged output file
    if total_files > 0:
        logging.info("\nMerging processed files...")
        merged_file = merge_output_files(output_dir)
        if merged_file:
            logging.info(f"Successfully created merged file: {merged_file}")
    
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
    
    logging.info(f"\nProcessing complete. Log file: {log_file}")

# Example usage
if __name__ == "__main__":
    input_directory = "bank_statements"
    output_directory = "processed_statements"
    mapping_rules_file = "mapping-rules.csv"

    process_directory(input_directory, output_directory, mapping_rules_file)