"""
Configuration classes and enums for Delta Lake datasource.

This module contains configuration data classes and enums for Delta Lake
write operations using two-phase commit for ACID compliance.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class WriteMode(Enum):
    """Write modes for Delta Lake tables."""
    # Error if the table already exists
    ERROR = "error"
    # Append to the table if it exists
    APPEND = "append"
    # Overwrite the table if it exists
    OVERWRITE = "overwrite"
    # Ignore the write if the table already exists
    IGNORE = "ignore"










class DeltaJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, bytes):
            return obj.decode("unicode_escape", "backslashreplace")
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


@dataclass
class DeltaWriteConfig:
    """Configuration for Delta Lake write operations."""

    mode: WriteMode = WriteMode.APPEND
    partition_cols: Optional[List[str]] = None
    schema_mode: str = "merge"
    name: Optional[str] = None
    description: Optional[str] = None
    configuration: Optional[Dict[str, str]] = None
    custom_metadata: Optional[Dict[str, str]] = None
    target_file_size: Optional[int] = None
    writer_properties: Optional[Any] = None  # deltalake.WriterProperties
    post_commithook_properties: Optional[Any] = None  # deltalake.PostCommitHookProperties
    commit_properties: Optional[Any] = None  # deltalake.CommitProperties
    storage_options: Optional[Dict[str, str]] = None
    engine: str = "rust"
    overwrite_schema: bool = False
    schema: Optional[Any] = None  # pyarrow.Schema
