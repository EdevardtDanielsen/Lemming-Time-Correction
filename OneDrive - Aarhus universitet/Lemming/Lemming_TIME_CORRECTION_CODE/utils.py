# =============================================================================
# Utility Functions and Helpers
# =============================================================================
# Common operations for working with aligned data.
#
# Why this exists: Provides reusable functions for loading, analyzing, and
# validating aligned data. Used for post-processing, quality checks, and
# multi-device merging.
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


# Load aligned data with proper datetime parsing
# Returns DataFrame with datetime columns properly converted
def load_aligned_data(aligned_data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(aligned_data_path)
    
    # Ensure datetime columns are proper datetime
    for col in ['original_rtc', 'aligned_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


# Load a reconstructed .TXT file with timestamp parsing
# expected_columns defaults to standard sensor columns if not provided
def load_reconstructed_file(file_path: Path,
                           expected_columns: list = None) -> pd.DataFrame:
    if expected_columns is None:
        expected_columns = [
            'CH4_1', 'CH4_2', 'CO2', 'RH', 'Temp', 'Pressure', 'timestamp'
        ]
    
    df = pd.read_csv(file_path, names=expected_columns, header=None)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')
    
    return df


# Compare performance of different alignment methods
# Returns summary DataFrame grouped by method with confidence and drift stats
def compare_alignment_methods(aligned_df: pd.DataFrame) -> pd.DataFrame:
    if 'method' not in aligned_df.columns:
        logger.warning("No 'method' column found in aligned data")
        return pd.DataFrame()
    
    aligned_df['drift_seconds'] = (
        aligned_df['aligned_time'] - aligned_df['original_rtc']
    ).dt.total_seconds()
    
    summary = aligned_df.groupby('method').agg({
        'confidence': ['mean', 'min', 'max', 'count'],
        'drift_seconds': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    return summary


# Identify segments with low alignment quality (below min_confidence threshold)
# Returns DataFrame with problematic segments sorted by confidence
def find_problematic_segments(aligned_df: pd.DataFrame,
                             min_confidence: float = 0.6) -> pd.DataFrame:
    if 'segment_id' not in aligned_df.columns:
        logger.warning("No 'segment_id' column found")
        return pd.DataFrame()
    
    segment_stats = aligned_df.groupby('segment_id').agg({
        'confidence': 'mean',
        'original_rtc': ['min', 'max'],
        'aligned_time': ['min', 'max']
    }).reset_index()
    
    segment_stats.columns = [
        'segment_id', 'mean_confidence', 
        'rtc_start', 'rtc_end', 'ttn_start', 'ttn_end'
    ]
    
    problematic = segment_stats[segment_stats['mean_confidence'] < min_confidence]
    
    return problematic.sort_values('mean_confidence')


# Export only high-confidence aligned data to parquet file
# Filters to records with confidence >= min_confidence threshold
def export_high_confidence_subset(aligned_df: pd.DataFrame,
                                  output_path: Path,
                                  min_confidence: float = 0.9):
    high_conf = aligned_df[aligned_df['confidence'] >= min_confidence].copy()
    
    logger.info(f"Exporting {len(high_conf)} / {len(aligned_df)} high-confidence records")
    logger.info(f"Percentage: {len(high_conf) / len(aligned_df) * 100:.1f}%")
    
    high_conf.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")


# Generate comprehensive data quality metrics dictionary
# Includes confidence stats, drift stats, segment info, and method breakdown
def generate_data_quality_report(aligned_df: pd.DataFrame) -> dict:
    aligned_df['drift_seconds'] = (
        aligned_df['aligned_time'] - aligned_df['original_rtc']
    ).dt.total_seconds()
    
    report = {
        'total_records': len(aligned_df),
        'date_range': {
            'rtc': (
                str(aligned_df['original_rtc'].min()),
                str(aligned_df['original_rtc'].max())
            ),
            'aligned': (
                str(aligned_df['aligned_time'].min()),
                str(aligned_df['aligned_time'].max())
            )
        },
        'confidence': {
            'mean': float(aligned_df['confidence'].mean()),
            'median': float(aligned_df['confidence'].median()),
            'std': float(aligned_df['confidence'].std()),
            'min': float(aligned_df['confidence'].min()),
            'max': float(aligned_df['confidence'].max()),
            'high_pct': float((aligned_df['confidence'] >= 0.9).mean() * 100),
            'medium_pct': float(
                ((aligned_df['confidence'] >= 0.6) & 
                 (aligned_df['confidence'] < 0.9)).mean() * 100
            ),
            'low_pct': float((aligned_df['confidence'] < 0.6).mean() * 100)
        },
        'drift': {
            'mean_seconds': float(aligned_df['drift_seconds'].mean()),
            'median_seconds': float(aligned_df['drift_seconds'].median()),
            'std_seconds': float(aligned_df['drift_seconds'].std()),
            'min_seconds': float(aligned_df['drift_seconds'].min()),
            'max_seconds': float(aligned_df['drift_seconds'].max()),
            'range_hours': float(
                (aligned_df['drift_seconds'].max() - 
                 aligned_df['drift_seconds'].min()) / 3600
            )
        }
    }
    
    if 'segment_id' in aligned_df.columns:
        report['segments'] = {
            'total': int(aligned_df['segment_id'].nunique()),
            'mean_size': float(aligned_df.groupby('segment_id').size().mean()),
            'median_size': float(aligned_df.groupby('segment_id').size().median())
        }
    
    if 'method' in aligned_df.columns:
        report['methods'] = aligned_df['method'].value_counts().to_dict()
    
    return report


# Check time coverage of reconstructed files against expected range
# Returns DataFrame showing gaps where hourly files are missing
def check_file_coverage(reconstructed_dir: Path,
                       expected_start: datetime,
                       expected_end: datetime) -> pd.DataFrame:
    # Get all hourly files
    txt_files = list(reconstructed_dir.rglob("*.TXT"))
    
    if len(txt_files) == 0:
        logger.warning("No files found")
        return pd.DataFrame()
    
    # Parse filenames to get hour keys
    hours = []
    for f in txt_files:
        try:
            # Filename format: YYMMDDHH.TXT
            hour_key = f.stem  # Remove .TXT
            dt = datetime.strptime(hour_key, '%y%m%d%H')
            hours.append(dt)
        except:
            logger.debug(f"Could not parse filename: {f.name}")
    
    hours = sorted(hours)
    
    if len(hours) == 0:
        logger.warning("No valid hour files found")
        return pd.DataFrame()
    
    # Find gaps
    expected_hours = pd.date_range(
        start=expected_start.replace(minute=0, second=0, microsecond=0),
        end=expected_end.replace(minute=0, second=0, microsecond=0),
        freq='H'
    )
    
    actual_hours = pd.DatetimeIndex(hours)
    missing_hours = expected_hours.difference(actual_hours)
    
    if len(missing_hours) == 0:
        logger.info("[OK] Full coverage - no missing hours")
        return pd.DataFrame({'status': ['complete']})
    
    # Group consecutive missing hours into gaps
    gaps = []
    if len(missing_hours) > 0:
        gap_start = missing_hours[0]
        gap_end = missing_hours[0]
        
        for i in range(1, len(missing_hours)):
            if (missing_hours[i] - gap_end).total_seconds() <= 3600:
                gap_end = missing_hours[i]
            else:
                gaps.append({
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'duration_hours': int((gap_end - gap_start).total_seconds() / 3600) + 1
                })
                gap_start = missing_hours[i]
                gap_end = missing_hours[i]
        
        # Add final gap
        gaps.append({
            'gap_start': gap_start,
            'gap_end': gap_end,
            'duration_hours': int((gap_end - gap_start).total_seconds() / 3600) + 1
        })
    
    gaps_df = pd.DataFrame(gaps)
    
    logger.info(f"Found {len(gaps)} gaps in coverage")
    logger.info(f"Total missing hours: {len(missing_hours)}")
    
    return gaps_df


# Merge aligned data from multiple devices into a single parquet file
# Adds device_id column to identify source device
def merge_multiple_devices(device_dirs: list,
                          output_path: Path,
                          device_ids: list = None):
    all_data = []
    
    for i, device_dir in enumerate(device_dirs):
        aligned_file = device_dir / "aligned_segments" / "aligned_data.parquet"
        
        if not aligned_file.exists():
            logger.warning(f"No aligned data found for device {i}")
            continue
        
        df = pd.read_parquet(aligned_file)
        
        # Add device identifier
        if device_ids and i < len(device_ids):
            df['device_id'] = device_ids[i]
        else:
            df['device_id'] = f"device_{i}"
        
        all_data.append(df)
        logger.info(f"Loaded {len(df)} records from device {df['device_id'].iloc[0]}")
    
    if len(all_data) == 0:
        logger.error("No data loaded from any device")
        return None
    
    # Combine
    merged = pd.concat(all_data, ignore_index=True)
    merged = merged.sort_values('aligned_time')
    
    logger.info(f"Merged {len(merged)} total records from {len(all_data)} devices")
    
    # Save
    merged.to_parquet(output_path, index=False)
    logger.info(f"Saved merged data to {output_path}")
    
    return merged


# Create summary statistics for each hourly file (sample counts, timing, sensor means)
# Saves results to CSV at output_path
def create_hourly_summary_statistics(reconstructed_dir: Path,
                                     output_path: Path):
    txt_files = list(reconstructed_dir.rglob("*.TXT"))
    
    summaries = []
    
    for file_path in txt_files:
        try:
            df = load_reconstructed_file(file_path)
            
            summary = {
                'filename': file_path.name,
                'num_samples': len(df),
                'start_time': df['timestamp'].min(),
                'end_time': df['timestamp'].max(),
                'duration_minutes': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60,
                'mean_sample_interval': df['timestamp'].diff().dt.total_seconds().mean(),
                'max_gap_seconds': df['timestamp'].diff().dt.total_seconds().max()
            }
            
            # Add data statistics if columns exist
            for col in ['CH4_1', 'CH4_2', 'CO2', 'Temp', 'RH']:
                if col in df.columns:
                    summary[f'{col}_mean'] = df[col].mean()
                    summary[f'{col}_std'] = df[col].std()
            
            summaries.append(summary)
        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"Created hourly summary for {len(summary_df)} files")
    logger.info(f"Saved to {output_path}")
    
    return summary_df


if __name__ == "__main__":
    # Example usage
    import config
    
    logging.basicConfig(level=logging.INFO)
    
    # Load aligned data
    aligned_file = config.ALIGNED_DATA_DIR / "aligned_data.parquet"
    if aligned_file.exists():
        df = load_aligned_data(aligned_file)
        
        # Generate quality report
        report = generate_data_quality_report(df)
        
        print("\n=== DATA QUALITY REPORT ===")
        print(json.dumps(report, indent=2, default=str))
        
        # Check for problematic segments
        problematic = find_problematic_segments(df, min_confidence=0.7)
        if len(problematic) > 0:
            print(f"\n[WARNING] Found {len(problematic)} segments with confidence < 0.7")
            print(problematic)
        
        # Compare methods
        method_comparison = compare_alignment_methods(df)
        if not method_comparison.empty:
            print("\n=== ALIGNMENT METHOD COMPARISON ===")
            print(method_comparison)
    
    # Check file coverage
    if config.RECONSTRUCTED_DATA_DIR.exists():
        coverage = check_file_coverage(
            config.RECONSTRUCTED_DATA_DIR,
            datetime(2025, 1, 1),
            datetime(2025, 12, 31)
        )
        
        if not coverage.empty and 'status' not in coverage.columns:
            print("\n=== COVERAGE GAPS ===")
            print(coverage)
