# models.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path

@dataclass
class ProcessingConfig:
    """Configuration settings for bank statement processing"""
    required_columns: List[str] = field(default_factory=lambda: [
        'Category', 'SubCategory', 'Description', 
        'Debit', 'Credit', 'Balance', 'Date'
    ])
    output_columns: List[str] = field(default_factory=lambda: [
        'Category', 'SubCategory', 'Description', 
        'Amount', 'Date', 'Bank', 'Owner'
    ])
    max_field_length: int = 20
    date_formats: List[str] = field(default_factory=lambda: [
        r'\d{2}(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',
        r'\d{2}[-/]\d{2}[-/]\d{4}',
        r'\d{4}[-/]\d{2}[-/]\d{2}'
    ])
    time_formats: List[str] = field(default_factory=lambda: [
        r'\d{2}:\d{2}',
        r'\d{1,2}:\d{2}\s*(?:AM|PM)',
        r'\d{4}\s*HRS'
    ])

class TransactionStatus(Enum):
    """Status of transaction processing"""
    UNCHANGED = 'unchanged'
    MODIFIED = 'modified'
    ERROR = 'error'

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_files: int = 0
    total_records: int = 0
    modified_records: int = 0
    error_records: int = 0
    pattern_matches: Dict[str, int] = field(default_factory=dict)

    def update_from_file(self, file_stats: 'FileStats') -> None:
        """Update overall stats with single file statistics"""
        self.total_files += 1
        self.total_records += file_stats.total_records
        self.modified_records += file_stats.modified_records
        self.error_records += file_stats.error_records
        
        for pattern, count in file_stats.pattern_matches.items():
            self.pattern_matches[pattern] = (
                self.pattern_matches.get(pattern, 0) + count
            )

@dataclass
class FileStats:
    """Statistics for single file processing"""
    filename: str
    total_records: int = 0
    modified_records: int = 0
    error_records: int = 0
    pattern_matches: Dict[str, int] = field(default_factory=dict)

class BankStatementError(Exception):
    """Base exception for bank statement processing"""
    pass

class FileProcessingError(BankStatementError):
    """Error processing a specific file"""
    pass

class MappingRuleError(BankStatementError):
    """Error with mapping rules"""
    pass

class ValidationError(BankStatementError):
    """Error validating data"""
    pass