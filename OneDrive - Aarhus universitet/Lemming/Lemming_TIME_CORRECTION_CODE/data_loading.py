# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
# UNIVERSAL VERSION - Handles OLD (TTN) and NEW (Server) timestamp formats
# NOW WITH CONCATENATION RECOVERY - Handles missing newlines in SD card files
#
# Why this exists: SD card data and TTN reference data come in different formats.
# This module provides unified loaders that handle format detection and parsing
# automatically, including recovery of corrupted concatenated rows.
#
# Version: 2.0 with concatenation recovery
# Last updated: 2025-11-25
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Union
import re
import logging
from dateutil import parser
import ast
import pytz

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SDCardDataLoader: Load and preprocess SD card .TXT files
# RECURSIVE - Finds all .TXT files regardless of subfolder structure
# HANDLES CONCATENATED ROWS - Recovers data from missing newlines (SD write failures)
#
# Concatenation types handled:
#   Type 1: Different timestamps on same line -> Split and keep both
#   Type 2: Same timestamp, different values -> Split and spread duplicates
#   Type 3: Identical rows -> Split and deduplicate later
# -----------------------------------------------------------------------------
class SDCardDataLoader:

    def __init__(self, data_dir: Path, rtc_format: str, rtc_offset_minutes: int = 60):
        self.data_dir = data_dir
        self.rtc_format = rtc_format
        # CRITICAL: The device RTC was set using Arduino __TIME__ at compile time.
        # It stores a FIXED offset from UTC (no DST awareness).
        # Default is UTC+1 (60 minutes) for devices compiled in Copenhagen winter.
        self.rtc_offset_minutes = rtc_offset_minutes
        self.rtc_timezone = pytz.FixedOffset(rtc_offset_minutes)
        self.files_loaded = 0
        self.lines_loaded = 0
        self.lines_skipped = 0
        
        # NEW: Concatenation recovery tracking
        self.concatenated_lines_found = 0
        self.rows_recovered = 0
        self.recovery_by_type = {'type1': 0, 'type2': 0, 'type3': 0}
    
    # Load all .TXT files from SD card directory (recursive search)
    # Returns DataFrame with: file_path, line_number, line_content, rtc_time,
    #                        data_values, is_concatenated, concat_position
    def load_all_txt_files(self) -> pd.DataFrame:
        logger.info(f"Scanning for .TXT files in {self.data_dir} (recursive)")
        
        # Recursive glob - finds files in any subdirectory
        txt_files = list(self.data_dir.rglob("*.TXT"))
        logger.info(f"Found {len(txt_files)} .TXT files")
        
        if len(txt_files) > 0:
            logger.info("File locations:")
            for f in txt_files[:5]:  # Show first 5
                rel_path = f.relative_to(self.data_dir)
                logger.info(f"  {rel_path}")
            if len(txt_files) > 5:
                logger.info(f"  ... and {len(txt_files) - 5} more")
        
        all_data = []
        
        for file_path in txt_files:
            file_data = self._load_single_file(file_path)
            if len(file_data) > 0:
                all_data.extend(file_data)
                self.files_loaded += 1
        
        if len(all_data) == 0:
            logger.warning("No data loaded from any files!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        logger.info(f"\nSuccessfully loaded:")
        logger.info(f"  - {self.files_loaded} files")
        logger.info(f"  - {self.lines_loaded:,} total rows")
        logger.info(f"  - {self.lines_skipped:,} lines skipped due to errors")
        
        # NEW: Report concatenation recovery
        if self.concatenated_lines_found > 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"[RECOVERY] Concatenated Row Detection & Recovery")
            logger.info(f"{'='*80}")
            logger.info(f"  Concatenated lines found: {self.concatenated_lines_found:,}")
            logger.info(f"  Total measurements recovered: {self.rows_recovered:,}")
            logger.info(f"  Average rows per concatenated line: {self.rows_recovered / self.concatenated_lines_found:.1f}")
            
            if sum(self.recovery_by_type.values()) > 0:
                logger.info(f"\n  Recovery by type:")
                logger.info(f"    Type 1 (Different timestamps): {self.recovery_by_type['type1']:,}")
                logger.info(f"    Type 2 (Same time, diff values): {self.recovery_by_type['type2']:,}")
                logger.info(f"    Type 3 (Identical duplicates): {self.recovery_by_type['type3']:,}")
            
            pct_recovered = (self.rows_recovered / self.lines_loaded) * 100
            logger.info(f"\n  Recovered data: {pct_recovered:.1f}% of total dataset")
            logger.info(f"{'='*80}")
        
        logger.info(f"\n  Date range: {df['rtc_time'].min()} to {df['rtc_time'].max()}")
        
        return df
    
    # Load a single .TXT file, parse contents, handle concatenated rows, deduplicate
    def _load_single_file(self, file_path: Path) -> List[dict]:
        file_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse line (may return multiple records if concatenated)
                    parsed_records = self._parse_line(line, file_path, line_num)
                    
                    if parsed_records is not None:
                        if isinstance(parsed_records, list):
                            # Concatenated line - multiple records recovered
                            file_data.extend(parsed_records)
                            self.lines_loaded += len(parsed_records)
                            self.concatenated_lines_found += 1
                            self.rows_recovered += len(parsed_records)
                        else:
                            # Normal line - single record
                            file_data.append(parsed_records)
                            self.lines_loaded += 1
                    else:
                        self.lines_skipped += 1
        
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {e}")
        
        # DEDUPLICATE: Remove duplicate measurements within this file
        if len(file_data) > 0:
            original_count = len(file_data)
            
            unique_data = []
            seen = set()
            
            for rec in file_data:
                # Create fingerprint: timestamp + data values + position/group to avoid dropping recovered splits
                fingerprint = (
                    rec['rtc_time'],
                    tuple(rec['data_values']),
                    rec.get('concat_position'),
                    rec.get('concat_group_id')
                )
                
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    unique_data.append(rec)
            
            duplicates_removed = original_count - len(unique_data)
            
            if duplicates_removed > 0:
                logger.debug(f"Removed {duplicates_removed} duplicates from {file_path.name} "
                            f"({duplicates_removed/original_count*100:.1f}%)")
            
            return unique_data
        
        return file_data
    
    # Parse a single line of SD card data (handles concatenated rows)
    # Expected format: CH4_1,CH4_2,CO2,RH,Temp,Pressure,DD.MM.YYYY HH:MM:SS
    # Detection: Count timestamp patterns - if >1, line is concatenated and needs splitting
    # Returns: Single dict (normal), List[dict] (concatenated), or None (invalid)
    def _parse_line(self, line: str, file_path: Path, line_num: int) -> Optional[Union[dict, List[dict]]]:
        try:
            # Detect concatenated rows by counting timestamp patterns
            timestamp_pattern = r'\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}'
            timestamps = re.findall(timestamp_pattern, line)
            
            if len(timestamps) > 1:
                # CONCATENATED LINE - split and parse each part
                return self._split_concatenated_line(line, timestamps, file_path, line_num)
            else:
                # NORMAL LINE - parse as before
                return self._parse_single_measurement(line, file_path, line_num)
        
        except Exception as e:
            logger.debug(f"Error parsing line in {file_path.name}:{line_num}: {e}")
            return None
    
    # Parse a normal (non-concatenated) line
    # Format: value1,value2,...,valueN,DD.MM.YYYY HH:MM:SS
    def _parse_single_measurement(self, line: str, file_path: Path, line_num: int) -> Optional[dict]:
        try:
            parts = line.split(',')
            
            if len(parts) < 2:
                return None
            
            # Last part should be timestamp
            timestamp_str = parts[-1].strip()
            
            # Parse RTC timestamp with FIXED timezone offset
            # The device RTC does NOT follow DST - it was set at compile time with a fixed offset
            try:
                rtc_time = datetime.strptime(timestamp_str, self.rtc_format)
                # Use fixed offset (e.g., UTC+1 for devices compiled in Copenhagen winter)
                rtc_time = self.rtc_timezone.localize(rtc_time)
                rtc_time = rtc_time.astimezone(timezone.utc)
                
            except ValueError:
                # Only log if safe (avoid crashing on binary data)
                if timestamp_str.isascii() and len(timestamp_str) < 100:
                    logger.debug(f"Could not parse timestamp '{timestamp_str}' in {file_path.name}:{line_num}")
                return None
            
            # Extract data values
            data_values = [p.strip() for p in parts[:-1]]
            
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'line_number': line_num,
                'line_content': line,
                'concat_group_id': f"{file_path.name}:{line_num}",  # stable id for original physical line
                'rtc_time': rtc_time,
                'timestamp_str': timestamp_str,
                'data_values': data_values,
                'num_fields': len(data_values),
                'is_concatenated': False,
                'concat_position': 0
            }
        
        except Exception as e:
            logger.debug(f"Error parsing measurement: {e}")
            return None
    
    # Split a concatenated line into individual measurements
    # Strategy: Find timestamp positions, split into segments, parse each segment
    # Returns list of parsed measurement dicts, or None if parsing fails
    def _split_concatenated_line(self, line: str, timestamps: List[str],
                                 file_path: Path, line_num: int) -> Optional[List[dict]]:
        records = []
        
        # Find all timestamp positions
        timestamp_positions = []
        for ts in timestamps:
            pos = line.find(ts)
            if pos != -1:
                timestamp_positions.append((pos, ts))
        
        # Sort by position (should already be sorted, but be safe)
        timestamp_positions.sort(key=lambda x: x[0])
        
        # Split line into segments
        for i, (ts_pos, ts_str) in enumerate(timestamp_positions):
            try:
                # Find start of this measurement
                if i == 0:
                    # First measurement starts at beginning of line
                    start = 0
                else:
                    # Subsequent measurements start after previous timestamp
                    prev_ts_pos, prev_ts_str = timestamp_positions[i-1]
                    start = prev_ts_pos + len(prev_ts_str)
                
                # End is at this timestamp + its length
                end = ts_pos + len(ts_str)
                
                # Extract segment
                segment = line[start:end].strip()

                # CHECK: Skip corrupted segments with non-ASCII data
                if not segment.isascii() or '\x00' in segment:
                    continue  # Skip silently

                # Remove leading comma if present (from previous row's data)
                if segment.startswith(','):
                    segment = segment[1:]

                # Parse segment
                parts = segment.split(',')          

                if len(parts) < 2:
                    logger.debug(f"Segment {i} has too few parts: {segment}")
                    continue
                
                # Timestamp should be last part
                segment_timestamp = parts[-1].strip()
                
                # Parse timestamp with FIXED timezone offset (no DST)
                rtc_time = datetime.strptime(segment_timestamp, self.rtc_format)
                rtc_time = self.rtc_timezone.localize(rtc_time)
                rtc_time = rtc_time.astimezone(timezone.utc)
                
                # Extract data values
                data_values = [p.strip() for p in parts[:-1]]
                
                records.append({
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'line_number': line_num,
                    'line_content': segment,  # Store segment, not full concatenated line
                    'concat_group_id': f"{file_path.name}:{line_num}",  # group all recovered measurements from this line
                    'rtc_time': rtc_time,
                    'timestamp_str': segment_timestamp,
                    'data_values': data_values,
                    'num_fields': len(data_values),
                    'is_concatenated': True,  # Mark as recovered from concatenation
                    'concat_position': i  # Position in original concatenated line
                })
            
            except Exception as e:
                logger.debug(f"Error parsing segment {i} in {file_path.name}:{line_num}: {e}")
                continue
        

        # Classify concatenation type (for statistics)
        if len(records) >= 2:
            self._classify_concatenation_type(records)

        # SORT by timestamp to fix interleaved data
        records.sort(key=lambda x: x['rtc_time'])

        return records if len(records) > 0 else None    
    
    # Classify concatenation type for statistics
    # Type 1: Different timestamps (99%+), Type 2: Same time/diff values, Type 3: Identical
    def _classify_concatenation_type(self, records: List[dict]) -> None:
        # Compare first two records
        rec1 = records[0]
        rec2 = records[1]
        
        if rec1['rtc_time'] != rec2['rtc_time']:
            # Type 1: Different timestamps
            self.recovery_by_type['type1'] += len(records)
        elif rec1['data_values'] != rec2['data_values']:
            # Type 2: Same timestamp, different values
            self.recovery_by_type['type2'] += len(records)
        else:
            # Type 3: Identical duplicates
            self.recovery_by_type['type3'] += len(records)


# -----------------------------------------------------------------------------
# DualFormatTTNLoader: Load TTN reference data from both OLD and NEW formats
# Automatically detects format based on file content and uses appropriate parser.
# OLD format: Tab-separated with 'received_at' header (from TTN console export)
# NEW format: Space-separated starting with timestamp (from server logs)
# -----------------------------------------------------------------------------
class DualFormatTTNLoader:

    def __init__(self, ttn_dir: Path):
        self.ttn_dir = ttn_dir
        self.old_loader = OldTTNLoader()
        self.new_loader = NewServerLoader()
    
    # Load TTN reference from all files in directory (both formats)
    # Returns DataFrame with: unix, received_at, format
    def load_ttn_reference(self) -> pd.DataFrame:
        logger.info(f"Loading TTN reference data from {self.ttn_dir}")
        
        if not self.ttn_dir.exists():
            raise FileNotFoundError(f"TTN directory not found: {self.ttn_dir}")
        
        # Find all .txt files
        txt_files = list(self.ttn_dir.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} .txt files in TTN directory")
        
        if len(txt_files) == 0:
            raise ValueError(f"No .txt files found in {self.ttn_dir}")
        
        all_ttn_data = []
        format_counts = {'OLD_TTN': 0, 'NEW_SERVER': 0, 'UNKNOWN': 0}
        
        for filepath in txt_files:
            # Detect format
            file_format = self._detect_format(filepath)
            format_counts[file_format] += 1
            
            logger.info(f"Processing {filepath.name} ({file_format})")
            
            # Load based on format
            if file_format == 'OLD_TTN':
                data = self.old_loader.load_file(filepath)
            elif file_format == 'NEW_SERVER':
                data = self.new_loader.load_file(filepath)
            else:
                logger.warning(f"Unknown format: {filepath.name}, skipping")
                continue
            
            if len(data) > 0:
                all_ttn_data.extend(data)
        
        if len(all_ttn_data) == 0:
            raise ValueError("No TTN data could be extracted from any files")
        
        # Combine and deduplicate
        df = pd.DataFrame(all_ttn_data)
        df = df.sort_values('unix').drop_duplicates('unix').reset_index(drop=True)
        
        logger.info(f"\n--- TTN Loading Summary ---")
        logger.info(f"Files by format:")
        for fmt, count in format_counts.items():
            if count > 0:
                logger.info(f"  {fmt}: {count} files")
        logger.info(f"Total TTN reference points: {len(df)}")
        logger.info(f"  Time range: {df['received_at'].min()} to {df['received_at'].max()}")
        logger.info(f"  Unix range: {df['unix'].min()} to {df['unix'].max()}")
        logger.info(f"  Duration: {(df['received_at'].max() - df['received_at'].min()).days} days")
        
        # Check for large gaps
        gaps = df['unix'].diff()
        max_gap = gaps.max()
        if max_gap > 3600:
            logger.warning(f"Large gap detected in TTN data: {max_gap} seconds ({max_gap/3600:.1f} hours)")
            large_gaps = gaps[gaps > 3600].sort_values(ascending=False)
            logger.warning(f"Number of gaps > 1 hour: {len(large_gaps)}")
        
        return df
    
    # Detect if file is OLD (TTN console) or NEW (server log) format
    def _detect_format(self, filepath: Path) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
                # OLD format has header with "received_at"
                if 'received_at' in first_line:
                    return 'OLD_TTN'
                # NEW format starts with timestamp
                elif re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', first_line):
                    return 'NEW_SERVER'
                else:
                    return 'UNKNOWN'
        except Exception as e:
            logger.error(f"Error detecting format for {filepath.name}: {e}")
            return 'UNKNOWN'


# -----------------------------------------------------------------------------
# OldTTNLoader: Parse OLD format TTN data (tab-separated with header)
# Format: Tab-separated CSV with 'received_at' and 'decoded_payload' columns
# -----------------------------------------------------------------------------
class OldTTNLoader:

    # Load OLD TTN format file, extract unix timestamps from decoded_payload
    def load_file(self, filepath: Path) -> List[dict]:
        ttn_data = []
        
        try:
            # Read tab-separated file
            df = pd.read_csv(filepath, sep='\t')
            
            if 'received_at' not in df.columns or 'decoded_payload' not in df.columns:
                logger.error(f"Missing required columns in {filepath.name}")
                return []
            
            for idx, row in df.iterrows():
                try:
                    # Extract server time (received_at)
                    server_time_str = row['received_at']
                    server_time = pd.to_datetime(server_time_str, utc=True)
                    
                    # Extract device unix from decoded_payload
                    payload_str = row['decoded_payload']
                    payload_dict = ast.literal_eval(payload_str)
                    device_unix = payload_dict.get('unix', None)
                    
                    if device_unix is not None:
                        ttn_data.append({
                            'unix': device_unix,
                            'received_at': server_time,
                            'format': 'OLD_TTN',
                            'source_file': filepath.name
                        })
                
                except Exception as e:
                    # Skip problematic rows
                    continue
            
            logger.info(f"  Loaded {len(ttn_data)} records from OLD format")
        
        except Exception as e:
            logger.error(f"Error loading OLD format from {filepath.name}: {e}")
        
        return ttn_data


# -----------------------------------------------------------------------------
# NewServerLoader: Parse NEW format server data (space-separated, no header)
# Handles both single-line and two-line formats:
#   Single-line: datetime and unix on same line (A1, A3)
#   Two-line: datetime on line 1, unix on line 2 (A2, B1, B2)
# Supports: type=ghg_v1, type=ghg_v2, type=ghg_v2_12, type=ghg_v2_18
# -----------------------------------------------------------------------------
class NewServerLoader:

    # Load NEW server format file, handle single-line and two-line variants
    def load_file(self, filepath: Path) -> List[dict]:
        ttn_data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                # Check if this line has type=ghg_* (decoded data line)
                if 'type=ghg' not in line:
                    i += 1
                    continue
                
                try:
                    # Parse current line
                    parts = line.split()
                    
                    # Extract server time (first two fields: date + time) - make UTC-aware
                    server_time_str = f"{parts[0]} {parts[1]}"
                    server_time = pd.to_datetime(server_time_str, utc=True)
                    
                    # Extract key=value pairs from current line
                    data_dict = {}
                    for part in parts[2:]:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            data_dict[key] = value
                    
                    # Try to get unix from current line
                    device_unix = data_dict.get('unix', None)
                    
                    # If unix not found on current line, check next line (two-line format)
                    if device_unix is None and (i + 1) < len(lines):
                        next_line = lines[i + 1].strip()
                        
                        # Check if next line also has the same timestamp (two-line format)
                        if next_line.startswith(parts[0]) and 'unix=' in next_line:
                            # Extract unix from next line
                            next_parts = next_line.split()
                            for part in next_parts:
                                if part.startswith('unix='):
                                    device_unix = part.split('=')[1]
                                    break
                    
                    # Convert unix to int
                    if device_unix is not None:
                        device_unix = int(device_unix)
                        
                        if device_unix > 0:
                            ttn_data.append({
                                'unix': device_unix,
                                'received_at': server_time,
                                'format': 'NEW_SERVER',
                                'source_file': filepath.name
                            })
                
                except Exception as e:
                    # Skip problematic lines
                    logger.debug(f"Error parsing line in {filepath.name}: {e}")
                    pass
                
                i += 1
            
            logger.info(f"  Loaded {len(ttn_data)} records from NEW format")
        
        except Exception as e:
            logger.error(f"Error loading NEW format from {filepath.name}: {e}")
        
        return ttn_data


# -----------------------------------------------------------------------------
# TTNDataLoader: LEGACY wrapper for backward compatibility
# Redirects to DualFormatTTNLoader - use DualFormatTTNLoader directly in new code
# -----------------------------------------------------------------------------
class TTNDataLoader:

    def __init__(self, ttn_path: Path):
        # If given a file, use its parent directory
        if ttn_path.is_file():
            self.ttn_dir = ttn_path.parent
        else:
            self.ttn_dir = ttn_path
        
        self.dual_loader = DualFormatTTNLoader(self.ttn_dir)
    
    # Load TTN reference (redirects to dual-format loader)
    def load_ttn_reference(self) -> pd.DataFrame:
        return self.dual_loader.load_ttn_reference()


# -----------------------------------------------------------------------------
# DataSegmenter: Split data into segments based on time gaps
# Segments are created when there's a large gap (device off) or time reversal.
# Why: RTC drift correction works best within continuous segments.
# -----------------------------------------------------------------------------
class DataSegmenter:

    def __init__(self, max_gap, min_segment_size: int = 5):
        self.max_gap = max_gap
        self.min_segment_size = min_segment_size
    
    # Split data into segments based on time gaps and backward time jumps
    # Returns list of DataFrames, one per continuous segment
    def segment_data(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        if len(df) == 0:
            return []
        
        # Sort by RTC time
        df = df.sort_values('rtc_time').reset_index(drop=True)
        
        # Find split points
        time_diffs = df['rtc_time'].diff()
        
        # Split on:
        # 1. Large time gaps
        # 2. Backward time jumps (time reversal)
        split_points = []
        
        for i in range(1, len(df)):
            time_diff = time_diffs.iloc[i]
            
            # Check for gap or backward jump
            if pd.notna(time_diff):
                if time_diff > self.max_gap or time_diff < pd.Timedelta(0):
                    split_points.append(i)
        
        # Create segments
        segments = []
        start_idx = 0
        
        for split_idx in split_points:
            segment = df.iloc[start_idx:split_idx].copy()
            if len(segment) >= self.min_segment_size:
                segment['segment_id'] = len(segments)
                segments.append(segment)
            start_idx = split_idx
        
        # Add final segment
        final_segment = df.iloc[start_idx:].copy()
        if len(final_segment) >= self.min_segment_size:
            final_segment['segment_id'] = len(segments)
            segments.append(final_segment)
        
        logger.info(f"Created {len(segments)} segments from {len(df)} data points")
        
        # Log segment statistics
        segment_sizes = [len(s) for s in segments]
        if segment_sizes:
            logger.info(f"  - Segment sizes: min={min(segment_sizes)}, "
                       f"max={max(segment_sizes)}, median={np.median(segment_sizes):.0f}")
        
        return segments


# -----------------------------------------------------------------------------
# validate_loaded_data: Validate loaded data and compute quality metrics
# Checks SD card data quality, TTN coverage, temporal overlap, and
# concatenation recovery statistics.
# Returns dictionary with validation results for logging and reporting.
# -----------------------------------------------------------------------------
def validate_loaded_data(sd_df: pd.DataFrame,
                        ttn_df: pd.DataFrame,
                        rtc_valid_range: Tuple[datetime, datetime]) -> dict:
    results = {
        'sd_card': {},
        'ttn': {},
        'overlap': {},
        'concatenation_recovery': {}  # NEW
    }
    
    # SD card validation
    if len(sd_df) > 0:
        rtc_times = sd_df['rtc_time']
        rtc_min, rtc_max = rtc_valid_range
        
        valid_rtc = (rtc_times >= rtc_min) & (rtc_times <= rtc_max)
        
        results['sd_card'] = {
            'total_lines': len(sd_df),
            'unique_files': sd_df['file_name'].nunique(),
            'date_range': (rtc_times.min(), rtc_times.max()),
            'valid_timestamps': valid_rtc.sum(),
            'invalid_timestamps': (~valid_rtc).sum(),
            'pct_valid': valid_rtc.mean() * 100,
            'duplicate_timestamps': len(sd_df) - len(sd_df.drop_duplicates(['rtc_time', 'file_name'])),
        }
        
        # NEW: Concatenation recovery statistics
        if 'is_concatenated' in sd_df.columns:
            concatenated_rows = sd_df[sd_df['is_concatenated'] == True]
            
            results['concatenation_recovery'] = {
                'total_recovered': len(concatenated_rows),
                'pct_of_total': (len(concatenated_rows) / len(sd_df) * 100) if len(sd_df) > 0 else 0,
                'affected_files': concatenated_rows['file_name'].nunique() if len(concatenated_rows) > 0 else 0
            }
            
            logger.info("\n[CONCATENATION RECOVERY VALIDATION]")
            logger.info(f"  Recovered rows: {len(concatenated_rows):,} ({len(concatenated_rows) / len(sd_df) * 100:.1f}% of total)")
            logger.info(f"  Affected files: {concatenated_rows['file_name'].nunique() if len(concatenated_rows) > 0 else 0}")
            
            # Check if recovered rows have valid timestamps
            if len(concatenated_rows) > 0:
                recovered_valid = (concatenated_rows['rtc_time'] >= rtc_min) & (concatenated_rows['rtc_time'] <= rtc_max)
                logger.info(f"  Valid timestamps in recovered data: {recovered_valid.sum():,} ({recovered_valid.mean() * 100:.1f}%)")
    
    # TTN validation
    if len(ttn_df) > 0:
        results['ttn'] = {
            'total_points': len(ttn_df),
            'date_range': (ttn_df['received_at'].min(), ttn_df['received_at'].max()),
            'unix_range': (ttn_df['unix'].min(), ttn_df['unix'].max()),
            'duplicates': len(ttn_df) - len(ttn_df.drop_duplicates('unix'))
        }
        
        # Format breakdown if available
        if 'format' in ttn_df.columns:
            format_counts = ttn_df['format'].value_counts()
            results['ttn']['format_breakdown'] = format_counts.to_dict()
    
    # Check overlap
    if len(sd_df) > 0 and len(ttn_df) > 0:
        sd_unix_times = pd.to_datetime(sd_df['rtc_time']).apply(lambda x: int(x.timestamp()))
        ttn_unix_times = ttn_df['unix']
        
        sd_min, sd_max = sd_unix_times.min(), sd_unix_times.max()
        ttn_min, ttn_max = ttn_unix_times.min(), ttn_unix_times.max()
        
        overlap_start = max(sd_min, ttn_min)
        overlap_end = min(sd_max, ttn_max)
        
        has_overlap = overlap_start <= overlap_end
        
        results['overlap'] = {
            'has_overlap': has_overlap,
            'sd_range_unix': (sd_min, sd_max),
            'ttn_range_unix': (ttn_min, ttn_max),
            'overlap_range_unix': (overlap_start, overlap_end) if has_overlap else None,
            'overlap_duration_days': (overlap_end - overlap_start) / 86400 if has_overlap else 0
        }
    
    return results
