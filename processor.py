# processor.py
import pandas as pd
import re
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .models import (
    ProcessingConfig,
    FileStats,
    ValidationError,
    FileProcessingError
)

class BankStatementProcessor:
    """Handles processing of bank statements and application of mapping rules"""
    
    def __init__(
        self,
        config: ProcessingConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame has required columns and proper data types"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
            
        numeric_columns = ['Debit', 'Credit', 'Balance']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column {col} must be numeric")
                
        return True
        
    def clean_description(
        self, 
        description: str,
        remove_dates: bool = True,
        remove_times: bool = True
    ) -> str:
        """Enhanced description cleaning with more patterns and better handling"""
        if not isinstance(description, str):
            return str(description)
            
        cleaned = description.strip()
        
        if remove_dates:
            for pattern in self.config.date_formats:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
                
        if remove_times:
            for pattern in self.config.time_formats:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
                
        cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
        
    def apply_mapping_rules(
        self,
        df: pd.DataFrame,
        mapping_rules: Dict[str, Any],
        file_stats: FileStats
    ) -> pd.DataFrame:
        """Apply mapping rules to DataFrame with better error handling"""
        try:
            processed_df = df.copy()
            
            processed_df['Description'] = processed_df['Description'].apply(
                self.clean_description
            )
            
            for idx, row in processed_df.iterrows():
                try:
                    processed_df.loc[idx] = self._apply_single_row_mapping(
                        row, mapping_rules, file_stats.pattern_matches
                    )
                except Exception as e:
                    self.logger.error(f"Error processing row {idx}: {e}")
                    file_stats.error_records += 1
                    
            return processed_df
            
        except Exception as e:
            raise FileProcessingError(f"Error applying mapping rules: {e}")

    def _apply_single_row_mapping(
        self,
        row: pd.Series,
        mapping_rules: Dict[str, Any],
        pattern_matches: Dict[str, int]
    ) -> pd.Series:
        """Apply mapping rules to a single row with better validation"""
        # Implementation as before...
        pass