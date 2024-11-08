# service.py
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
import pandas as pd

from models.py import (
    ProcessingConfig, 
    ProcessingStats, 
    FileStats,
    ValidationError,
    MappingRuleError,
    FileProcessingError
)
from .processor import BankStatementProcessor

class BankStatementProcessingService:
    """Main service for processing bank statements"""
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        mapping_rules_file: Union[str, Path],
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize the bank statement processing service.
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory for processed output files
            mapping_rules_file: Path to CSV file containing mapping rules
            config: Optional configuration settings
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mapping_rules_file = Path(mapping_rules_file)
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        self.processor = BankStatementProcessor(self.config, self.logger)
        self.stats = ProcessingStats()

    def process_all_statements(self) -> ProcessingStats:
        """
        Process all bank statements in the input directory.
        
        Returns:
            ProcessingStats: Statistics about the processing run
        
        Raises:
            FileProcessingError: If there are errors processing files
            MappingRuleError: If there are errors with mapping rules
        """
        try:
            # Ensure directories exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and validate mapping rules
            mapping_rules, rules_df = self._load_mapping_rules()
            if not mapping_rules:
                raise MappingRuleError("No mapping rules loaded")
                
            # Process each CSV file in the input directory
            for file_path in self.input_dir.glob('*.csv'):
                try:
                    self.logger.info(f"\nProcessing file: {file_path.name}")
                    
                    # Process single file
                    file_stats = self._process_single_file(
                        file_path,
                        mapping_rules
                    )
                    
                    # Update overall statistics
                    self.stats.update_from_file(file_stats)
                    
                    # Log individual file statistics
                    self._log_file_statistics(file_stats)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path.name}: {e}")
                    continue
                    
            # Update mapping rules with new statistics
            if rules_df is not None:
                self._update_mapping_rules(rules_df)
                
            # Log overall statistics
            self._log_overall_statistics()
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"Error in processing: {e}")
            raise

    def _process_single_file(
        self,
        file_path: Path,
        mapping_rules: Dict[str, Any]
    ) -> FileStats:
        """Process a single bank statement file"""
        file_stats = FileStats(filename=file_path.name)
        
        try:
            # Read input file
            df = pd.read_csv(file_path)
            
            # Validate DataFrame
            self.processor.validate_dataframe(df, self.config.required_columns)
            
            # Initialize statistics
            file_stats.total_records = len(df)
            
            # Process the file
            processed_df = self.processor.apply_mapping_rules(
                df, mapping_rules, file_stats
            )
            
            # Save processed file
            output_path = self.output_dir / f"Processed_{file_path.name}"
            processed_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Processed file saved to {output_path}")
            return file_stats
            
        except Exception as e:
            raise FileProcessingError(f"Error processing {file_path}: {e}")

    def _load_mapping_rules(self) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """Load and validate mapping rules from file"""
        try:
            rules_df = pd.read_csv(self.mapping_rules_file)
            
            # Validate required columns
            required_columns = {'Pattern', 'Priority'}
            missing_columns = required_columns - set(rules_df.columns)
            if missing_columns:
                raise ValidationError(f"Missing required columns in mapping rules: {missing_columns}")
                
            # Add statistics columns if they don't exist
            for col in ['CurrentRunMatches', 'TotalMatches']:
                if col not in rules_df.columns:
                    rules_df[col] = 0
                    
            # Reset CurrentRunMatches for new run
            rules_df['CurrentRunMatches'] = 0
            
            # Sort by Priority (lower number = higher priority)
            rules_df['Priority'] = rules_df['Priority'].fillna(float('inf'))
            rules_df = rules_df.sort_values('Priority', ascending=True)
            
            # Create rules dictionary
            rules_dict = {}
            for idx, row in rules_df.iterrows():
                if pd.isna(row['Pattern']):
                    self.logger.warning(f"Skipping rule at index {idx}: Pattern is null")
                    continue
                    
                rules_dict[str(row['Pattern'])] = {
                    'values': row.drop(['Pattern', 'Priority', 'CurrentRunMatches', 'TotalMatches']).to_dict(),
                    'index': idx
                }
                
            return rules_dict, rules_df
            
        except Exception as e:
            self.logger.error(f"Error loading mapping rules: {e}")
            raise MappingRuleError(f"Failed to load mapping rules: {e}")

    def _update_mapping_rules(self, rules_df: pd.DataFrame) -> None:
        """Update mapping rules with new statistics and save to file"""
        try:
            # Update CurrentRunMatches and TotalMatches
            for pattern, count in self.stats.pattern_matches.items():
                mask = rules_df['Pattern'] == pattern
                if mask.any():
                    rules_df.loc[mask, 'CurrentRunMatches'] = count
                    rules_df.loc[mask, 'TotalMatches'] += count
                    
            # Save updated rules
            rules_df.to_csv(self.mapping_rules_file, index=False)
            self.logger.info(f"Updated mapping rules saved to {self.mapping_rules_file}")
            
        except Exception as e:
            self.logger.error(f"Error updating mapping rules: {e}")
            raise MappingRuleError(f"Failed to update mapping rules: {e}")

    def _log_file_statistics(self, file_stats: FileStats) -> None:
        """Log statistics for a single file"""
        self.logger.info(f"File statistics for {file_stats.filename}:")
        self.logger.info(f"  Total records: {file_stats.total_records}")
        self.logger.info(f"  Records modified: {file_stats.modified_records}")
        self.logger.info(f"  Records with errors: {file_stats.error_records}")
        self.logger.info(f"  Records unchanged: {file_stats.total_records - file_stats.modified_records}")

    def _log_overall_statistics(self) -> None:
        """Log overall processing statistics"""
        self.logger.info("\nOverall Processing Statistics:")
        self.logger.info(f"Total files processed: {self.stats.total_files}")
        self.logger.info(f"Total records processed: {self.stats.total_records}")
        self.logger.info(f"Total records modified: {self.stats.modified_records}")
        self.logger.info(f"Total records with errors: {self.stats.error_records}")
        self.logger.info(f"Total records unchanged: {self.stats.total_records - self.stats.modified_records}")
        
        self.logger.info("\nPattern Match Statistics:")
        for pattern, count in self.stats.pattern_matches.items():
            self.logger.info(f"Pattern '{pattern}': {count} matches")


def main():
    """Main entry point for the bank statement processor"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize service
        service = BankStatementProcessingService(
            input_dir="bank_statements",
            output_dir="processed_statements",
            mapping_rules_file="mapping-rules.csv"
        )
        
        # Process all statements
        stats = service.process_all_statements()
        
        # Log overall results
        logging.info("\nProcessing Complete!")
        logging.info(f"Total files processed: {stats.total_files}")
        logging.info(f"Total records processed: {stats.total_records}")
        logging.info(f"Records modified: {stats.modified_records}")
        logging.info(f"Errors encountered: {stats.error_records}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()