import pandas as pd
import re
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def load_mapping_rules(csv_file: str) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Load mapping rules from a CSV file into a dictionary.
    Rules are sorted by Priority (lower number = higher Priority).
    
    Args:
        csv_file (str): Path to the CSV file containing mapping rules.
        
    Returns:
        tuple: (rules_dict, original_df) where
               rules_dict: Dictionary of rules with pattern as key
               original_df: Original DataFrame for updating, None if error occurs
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    try:
        # Verify required columns will be present
        required_columns = {'Pattern', 'Priority'}
        
        # Read CSV with explicit dtype for Priority to ensure proper sorting
        rules_df = pd.read_csv(csv_file, dtype={'Priority': int})
        
        # Validate required columns exist
        missing_columns = required_columns - set(rules_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Validate DataFrame is not empty
        if rules_df.empty:
            raise pd.errors.EmptyDataError("CSV file contains no data")
            
        # Add statistics columns if they don't exist
        stats_columns = ['CurrentRunMatches', 'TotalMatches']
        for col in stats_columns:
            if col not in rules_df.columns:
                rules_df[col] = 0
            else:
                # Ensure correct data type for statistics columns
                rules_df[col] = rules_df[col].fillna(0).astype(int)
        
        # Reset CurrentRunMatches for new run
        rules_df['CurrentRunMatches'] = 0
        
        # Sort by Priority and handle missing/invalid priorities
        rules_df['Priority'] = rules_df['Priority'].fillna(float('inf'))
        rules_df = rules_df.sort_values('Priority', ascending=True)
        
        # Create rules dictionary with error checking for Pattern column
        rules_dict = {}
        exclude_cols = {'Pattern', 'Priority', 'CurrentRunMatches', 'TotalMatches'}
        
        for idx, row in rules_df.iterrows():
            pattern = row['Pattern']
            if pd.isna(pattern):
                logging.warning(f"Skipping row {idx}: Pattern is null")
                continue
                
            rules_dict[str(pattern)] = {
                'values': row.drop(exclude_cols).to_dict(),
                'index': idx
            }
        
        return rules_dict, rules_df
        
    except FileNotFoundError as e:
        logging.error(f"CSV file not found: {csv_file}")
        return {}, None
    except pd.errors.EmptyDataError as e:
        logging.error(f"CSV file is empty: {csv_file}")
        return {}, None
    except Exception as e:
        logging.error(f"Error loading mapping rules from {csv_file}: {str(e)}")
        return {}, None
def update_mapping_statistics(
    rules_df: pd.DataFrame, 
    pattern_matches: Dict[str, int]
) -> Optional[pd.DataFrame]:
    """
    Update the statistics (CurrentRunMatches and TotalMatches) in the mapping rules DataFrame.
    
    Args:
        rules_df (pd.DataFrame): The original mapping rules DataFrame containing Pattern,
                                CurrentRunMatches, and TotalMatches columns
        pattern_matches (dict): Dictionary with patterns as keys and match counts as values
    
    Returns:
        pd.DataFrame: Updated DataFrame with new statistics
        None: If an error occurs during processing
    
    Raises:
        ValueError: If required columns are missing or if data types are incorrect
    """
    try:
        # Validate input parameters
        if rules_df is None or pattern_matches is None:
            raise ValueError("rules_df and pattern_matches cannot be None")
            
        if rules_df.empty:
            logging.warning("Empty rules DataFrame provided")
            return rules_df
            
        # Verify required columns exist
        required_columns = {'Pattern', 'CurrentRunMatches', 'TotalMatches'}
        missing_columns = required_columns - set(rules_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Create a copy to avoid modifying the original DataFrame
        updated_df = rules_df.copy()
        
        # Ensure numeric columns are properly typed
        updated_df['CurrentRunMatches'] = updated_df['CurrentRunMatches'].fillna(0).astype(int)
        updated_df['TotalMatches'] = updated_df['TotalMatches'].fillna(0).astype(int)
        
        # Track patterns that weren't found in the DataFrame
        unmatched_patterns = set()
        
        # Update statistics for each pattern
        for pattern, count in pattern_matches.items():
            # Validate count
            if not isinstance(count, (int, float)) or count < 0:
                logging.warning(f"Invalid count for pattern '{pattern}': {count}. Skipping.")
                continue
                
            # Convert count to int
            count = int(count)
            
            # Find matching rows
            mask = updated_df['Pattern'] == str(pattern)
            matches_found = mask.any()
            
            if matches_found:
                try:
                    # Update current run matches
                    updated_df.loc[mask, 'CurrentRunMatches'] = count
                    # Update total matches
                    updated_df.loc[mask, 'TotalMatches'] += count
                except Exception as e:
                    logging.error(f"Error updating statistics for pattern '{pattern}': {str(e)}")
            else:
                unmatched_patterns.add(pattern)
        
        # Log unmatched patterns
        if unmatched_patterns:
            logging.warning(f"Patterns not found in rules: {unmatched_patterns}")
        
        # Validate updates
        invalid_stats = (
            (updated_df['CurrentRunMatches'] < 0) | 
            (updated_df['TotalMatches'] < 0)
        ).any()
        
        if invalid_stats:
            logging.error("Invalid negative statistics detected after update")
            return rules_df  # Return original if validation fails
            
        return updated_df
        
    except Exception as e:
        logging.error(f"Error updating mapping statistics: {str(e)}")
        return rules_df  # Return original DataFrame in case of error
def validate_pattern_matches(pattern_matches: Dict[str, int]) -> bool:
    """
    Validate the pattern_matches dictionary format and values.
    
    Args:
        pattern_matches: Dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(pattern_matches, dict):
        return False
        
    return all(
        isinstance(k, str) and 
        isinstance(v, (int, float)) and 
        v >= 0 
        for k, v in pattern_matches.items()
    )
def save_mapping_rules(
    rules_df: pd.DataFrame, 
    csv_file: Union[str, Path],
    create_backup: bool = True,
    backup_suffix: str = '.bak'
) -> bool:
    """
    Save updated mapping rules back to CSV file with validation and backup options.
    
    Args:
        rules_df (pd.DataFrame): The updated mapping rules DataFrame
        csv_file (str or Path): Path to the CSV file
        create_backup (bool): Whether to create a backup of existing file
        backup_suffix (str): Suffix to use for backup file
    
    Returns:
        bool: True if save was successful, False otherwise
    
    Raises:
        ValueError: If input parameters are invalid
        OSError: If file operations fail
    """
    try:
        # Validate input DataFrame
        if rules_df is None or not isinstance(rules_df, pd.DataFrame):
            raise ValueError("Invalid rules_df: Must provide a valid pandas DataFrame")
            
        if rules_df.empty:
            raise ValueError("Cannot save empty DataFrame")
            
        # Validate required columns
        required_columns = {'Pattern', 'Priority', 'CurrentRunMatches', 'TotalMatches'}
        missing_columns = required_columns - set(rules_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert string path to Path object
        file_path = Path(csv_file)
        
        # Validate file path
        if not file_path.parent.exists():
            raise OSError(f"Directory does not exist: {file_path.parent}")
            
        # Create backup if requested and file exists
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(backup_suffix)
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                logging.info(f"Backup created at {backup_path}")
            except Exception as e:
                logging.error(f"Failed to create backup: {e}")
                # Continue with save operation even if backup fails
                
        # Validate data types before saving
        try:
            rules_df['CurrentRunMatches'] = rules_df['CurrentRunMatches'].fillna(0).astype(int)
            rules_df['TotalMatches'] = rules_df['TotalMatches'].fillna(0).astype(int)
            rules_df['Priority'] = rules_df['Priority'].fillna(float('inf')).astype(int)
        except Exception as e:
            raise ValueError(f"Error converting data types: {e}")
            
        # Sort DataFrame by Priority before saving
        rules_df = rules_df.sort_values('Priority', ascending=True)
        
        # Attempt to save with temporary file
        temp_file = file_path.with_suffix('.tmp')
        try:
            # Save to temporary file first
            rules_df.to_csv(temp_file, index=False)
            
            # Verify the saved file
            verification_df = pd.read_csv(temp_file)
            if not verification_df.equals(rules_df):
                raise ValueError("Verification failed: Saved data does not match original")
                
            # Replace original file with temporary file
            if file_path.exists():
                file_path.unlink()  # Remove existing file
            temp_file.rename(file_path)
            
            logging.info(f"Successfully saved mapping rules to {file_path}")
            return True
            
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()  # Clean up temporary file
            raise OSError(f"Error during file save operation: {e}")
            
    except Exception as e:
        logging.error(f"Error saving mapping rules to {csv_file}: {str(e)}")
        return False
        
def validate_csv_path(file_path: Union[str, Path]) -> Optional[Path]:
    """
    Validate the CSV file path.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Path object if valid, None otherwise
    """
    try:
        path = Path(file_path)
        
        # Check if path is absolute
        if not path.is_absolute():
            path = path.resolve()
            
        # Validate directory exists
        if not path.parent.exists():
            return None
            
        # Validate file extension
        if path.suffix.lower() != '.csv':
            return None
            
        return path
        
    except Exception:
        return None

def create_file_backup(file_path: Path, backup_suffix: str = '.bak') -> bool:
    """
    Create a backup of the specified file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix for backup file
        
    Returns:
        bool: True if backup was successful, False otherwise
    """
    try:
        if not file_path.exists():
            return False
            
        backup_path = file_path.with_suffix(backup_suffix)
        import shutil
        shutil.copy2(file_path, backup_path)
        return True
        
    except Exception as e:
        logging.error(f"Backup creation failed: {e}")
        return False

def clean_description(
    description: Optional[Union[str, float]], 
    remove_dates: bool = True,
    remove_times: bool = True,
    remove_special_chars: bool = True,
    max_length: Optional[int] = None
) -> str:
    """
    Clean description text by removing dates, times, and normalizing formatting.
    
    Args:
        description (str or float or None): The description text to clean
        remove_dates (bool): Whether to remove date patterns
        remove_times (bool): Whether to remove time patterns
        remove_special_chars (bool): Whether to remove special characters
        max_length (int, optional): Maximum length for returned string
        
    Returns:
        str: Cleaned description text
        
    Examples:
        >>> clean_description("Meeting on 22JAN at 15:30")
        "Meeting on at"
        >>> clean_description("No date here!", remove_dates=False)
        "No date here!"
    """
    try:
        # Handle None or empty input
        if description is None or description == '':
            return ''
            
        # Convert float/int to string if needed
        if isinstance(description, (float, int)):
            description = str(description)
            
        # Ensure input is string
        if not isinstance(description, str):
            logging.warning(f"Invalid input type: {type(description)}. Converting to string.")
            description = str(description)
        
        # Initialize cleaned text
        cleaned = description.strip()
        
        if remove_dates:
            # Extended date patterns
            date_patterns = [
                # DDMMM format (e.g., 22JAN)
                r'\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',
                # DD-MM-YYYY format
                r'\d{2}[-/]\d{2}[-/]\d{4}',
                # YYYY-MM-DD format
                r'\d{4}[-/]\d{2}[-/]\d{2}',
                # Written month formats
                r'\d{1,2}\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)',
                r'\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
                # Additional date formats can be added here
            ]
            
            for pattern in date_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        if remove_times:
            # Extended time patterns
            time_patterns = [
                # HH:MM format
                r'\d{1,2}:\d{2}(?:\s*(?:AM|PM))?',
                # HH.MM format
                r'\d{1,2}\.\d{2}(?:\s*(?:AM|PM))?',
                # Military time
                r'\d{4}\s*(?:HRS|hrs)',
                # Written time formats
                r'\d{1,2}\s*(?:AM|PM)',
                # Additional time formats can be added here
            ]
            
            for pattern in time_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        if remove_special_chars:
            # Remove special characters but keep basic punctuation
            cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', cleaned)
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove multiple punctuation
        cleaned = re.sub(r'([.,!?])\1+', r'\1', cleaned)
        
        # Truncate if max_length is specified
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + '...'
        
        return cleaned.strip()
        
    except Exception as e:
        logging.error(f"Error cleaning description: {str(e)}")
        return description  # Return original text in case of error

def is_valid_date_format(text: str, formats: Optional[List[str]] = None) -> bool:
    """
    Check if text contains a valid date in any of the specified formats.
    
    Args:
        text: Text to check
        formats: List of date formats to check against
        
    Returns:
        bool: True if valid date found, False otherwise
    """
    if formats is None:
        formats = [
            '%d%b',  # 22JAN
            '%d-%m-%Y',  # 22-01-2024
            '%Y-%m-%d',  # 2024-01-22
            '%d/%m/%Y',  # 22/01/2024
        ]
    
    for fmt in formats:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    
    return False

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text to normalize
        
    Returns:
        str: Text with normalized whitespace
    """
    # Replace various whitespace characters with single space
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    # Fix spacing around punctuation
    normalized = re.sub(r'\s([.,!?])', r'\1', normalized)
    normalized = re.sub(r'([.,!?])\s', r'\1 ', normalized)
    
    return normalized.strip()
import pandas as pd
import re
import os
import logging
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

def apply_mapping(
    row: pd.Series,
    mapping_rules: Dict,
    pattern_matches: Dict[str, int],
    case_sensitive: bool = False,
    debug_mode: bool = False
) -> pd.Series:
    """
    Apply mapping rules to a single row of transaction data.
    
    Args:
        row (pd.Series): A row of transaction data
        mapping_rules (dict): Dictionary of mapping rules
        pattern_matches (dict): Dictionary to track pattern matches
        case_sensitive (bool): Whether pattern matching should be case sensitive
        debug_mode (bool): Enable detailed debug logging
        
    Returns:
        pd.Series: Updated row after applying mapping rules
        
    Raises:
        KeyError: If required columns are missing
        ValueError: If input data is invalid
    """
    try:
        # Validate input row
        required_columns = {'Description', 'Category', 'SubCategory', 'Owner', 'Bank'}
        missing_columns = required_columns - set(row.index)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
            
        # Store original values for logging
        original_values = {
            'Description': row['Description'],
            'Category': row['Category'],
            'SubCategory': row['SubCategory']
        }
        
        # Skip if already processed
        if row.get('changed', False):
            if debug_mode:
                logging.debug(f"Skipping already processed row: {row['Description']}")
            return row
            
        # Create copy of row to avoid modifying original
        updated_row = row.copy()
        
        # Clean description
        try:
            cleaned_description = clean_description(
                updated_row['Description'],
                remove_dates=True,
                remove_times=True,
                remove_special_chars=True
            )
            updated_row['Description'] = cleaned_description
        except Exception as e:
            logging.error(f"Error cleaning description: {e}")
            cleaned_description = updated_row['Description']
            
        # Prepare text for pattern matching
        combined_text = " ".join(filter(None, [
            str(updated_row.get('Category', '')),
            str(updated_row.get('SubCategory', '')),
            cleaned_description,
            str(updated_row.get('Owner', '')),
            str(updated_row.get('Bank', ''))
        ]))
        
        # Track applied rules for logging
        applied_rules = []
        
        # Apply mapping rules
        for pattern, rule_info in mapping_rules.items():
            try:
                # Prepare regex flags
                flags = 0 if case_sensitive else re.IGNORECASE
                
                if re.search(pattern, combined_text, flags=flags):
                    # Apply rule values
                    for key, value in rule_info['values'].items():
                        if pd.notna(value) and key in updated_row.index:
                            if key in ['Category', 'SubCategory', 'Description']:
                                updated_row[key] = str(value)
                                
                    # Mark as changed
                    updated_row['changed'] = True
                    
                    # Update pattern matches counter
                    pattern_matches[pattern] = pattern_matches.get(pattern, 0) + 1
                    
                    # Record applied rule
                    applied_rules.append({
                        'pattern': pattern,
                        'changes': {
                            k: v for k, v in rule_info['values'].items()
                            if k in ['Category', 'SubCategory', 'Description']
                        }
                    })
                    
                    # Break after first matching rule
                    break
                    
            except re.error as e:
                logging.error(f"Invalid regex pattern '{pattern}': {e}")
                continue
                
        # Log changes if debug mode is enabled
        if debug_mode and applied_rules:
            changes = []
            for field in ['Description', 'Category', 'SubCategory']:
                if original_values[field] != updated_row[field]:
                    changes.append(f"{field}: '{original_values[field]}' -> '{updated_row[field]}'")
                    
            if changes:
                logging.debug(
                    f"Applied rules: {applied_rules}\n"
                    f"Changes made: {', '.join(changes)}"
                )
                
        return updated_row
        
    except Exception as e:
        logging.error(f"Error applying mapping: {str(e)}")
        return row  # Return original row in case of error

def extract_bank_owner(
    filename: Union[str, Path],
    max_length: int = 20,
    default_values: Tuple[str, str] = ('Unknown', 'Unknown')
) -> Tuple[str, str]:
    """
    Extract bank and owner information from filename with validation and cleaning.
    
    Args:
        filename (str or Path): Input filename
        max_length (int): Maximum length for bank and owner fields
        default_values (tuple): Default (bank, owner) if extraction fails
        
    Returns:
        tuple: (bank, owner) - extracted and cleaned values
        
    Example:
        >>> extract_bank_owner("CHASE John Smith.csv")
        ('CHASE', 'John Smith')
    """
    try:
        # Convert to Path object
        file_path = Path(filename)
        
        # Validate input
        if not file_path.stem:
            raise ValueError("Empty filename")
            
        # Remove file extension and split by first space
        name_parts = file_path.stem.split(' ', 1)
        
        # Extract and clean bank name
        bank = (
            clean_text(name_parts[0], max_length)
            if name_parts
            else default_values[0]
        )
        
        # Extract and clean owner name
        owner = (
            clean_text(name_parts[1], max_length)
            if len(name_parts) > 1
            else default_values[1]
        )
        
        return bank, owner
        
    except Exception as e:
        logging.error(f"Error extracting bank/owner from filename '{filename}': {e}")
        return default_values

def clean_text(
    text: str,
    max_length: int = 20,
    remove_special_chars: bool = True
) -> str:
    """
    Clean and normalize text values.
    
    Args:
        text (str): Text to clean
        max_length (int): Maximum length for returned string
        remove_special_chars (bool): Whether to remove special characters
        
    Returns:
        str: Cleaned text
    """
    try:
        # Convert to string and strip whitespace
        cleaned = str(text).strip()
        
        # Remove special characters if requested
        if remove_special_chars:
            cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', cleaned)
            
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Truncate to max length if needed
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(' ', 1)[0]
            
        return cleaned
        
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text  # Return original text in case of error

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