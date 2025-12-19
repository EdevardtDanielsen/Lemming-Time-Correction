# =============================================================================
# ULTIMATE VALIDATION SCRIPT
# =============================================================================
# Comprehensive deep-dive validation of the entire RTC-TTN alignment pipeline.
# Works for any device (H1, C3, A1, B1, etc.)
#
# Why this exists: After the pipeline runs, we need to verify output quality.
# This script checks for: chronological order, duplicate timestamps, gaps,
# interval consistency, and cross-validates parquet vs file outputs.
#
# Created by: Edevardt Johan Danielsen (ED)
# Last edited: 18/11/2025
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import defaultdict
import re

# Import config to get device-specific paths
import config

print("=" * 100)
print("ULTIMATE PIPELINE VALIDATION - COMPREHENSIVE QUALITY CHECK")
print("=" * 100)
print(f"Device: {config.DEVICE_ID}")
print("\nThis will perform an exhaustive check of all pipeline outputs.")
print("Estimated time: 2-3 minutes\n")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = config.OUTPUT_DIR
RECONSTRUCTED_DIR = config.RECONSTRUCTED_DATA_DIR

# Expected parameters
EXPECTED_INTERVAL_SECONDS = config.DEVICE_SAMPLE_INTERVAL_SECONDS  # Device-specific!
TOLERANCE_SECONDS = 0.1  # Allow small floating point errors
MAX_ACCEPTABLE_GAP_MINUTES = 10  # Flag gaps larger than this

# Results tracking
validation_results = {
    'device_id': config.DEVICE_ID,
    'critical_issues': [],
    'warnings': [],
    'info': [],
    'statistics': {}
}

# Add a validation issue to results and print with appropriate icon
def add_issue(level, message):
    if level == 'critical':
        validation_results['critical_issues'].append(message)
        print(f"❌ CRITICAL: {message}")
    elif level == 'warning':
        validation_results['warnings'].append(message)
        print(f"⚠️  WARNING: {message}")
    else:
        validation_results['info'].append(message)
        print(f"ℹ️  INFO: {message}")

# =============================================================================
# SECTION 1: VALIDATE ALIGNED PARQUET FILE
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 1: ALIGNED DATA (PARQUET) VALIDATION")
print("=" * 100)

aligned_file = config.ALIGNED_DATA_DIR / "aligned_data.parquet"
print(f"\n📂 Loading: {aligned_file}")

if not aligned_file.exists():
    add_issue('critical', f"Aligned data file not found: {aligned_file}")
    print("\n❌ Cannot proceed without aligned data file!")
    exit(1)

try:
    df_aligned = pd.read_parquet(aligned_file)
    print(f"✅ Loaded {len(df_aligned):,} rows")
    validation_results['statistics']['total_aligned_rows'] = len(df_aligned)
except Exception as e:
    add_issue('critical', f"Cannot load aligned data: {e}")
    exit(1)

# Check 1.1: Required columns exist
print("\n--- Check 1.1: Required Columns ---")
required_cols = ['aligned_time', 'original_rtc', 'confidence', 'segment_id', 'data_values']
missing_cols = [col for col in required_cols if col not in df_aligned.columns]
if missing_cols:
    add_issue('critical', f"Missing required columns: {missing_cols}")
else:
    print("✅ All required columns present")

# Check 1.2: Data types
print("\n--- Check 1.2: Data Types ---")
if not pd.api.types.is_datetime64_any_dtype(df_aligned['aligned_time']):
    add_issue('critical', "aligned_time is not datetime type")
if not pd.api.types.is_datetime64_any_dtype(df_aligned['original_rtc']):
    add_issue('critical', "original_rtc is not datetime type")
if not pd.api.types.is_float_dtype(df_aligned['confidence']) and not pd.api.types.is_integer_dtype(df_aligned['confidence']):
    add_issue('warning', "confidence is not numeric type")
print("✅ Data types validated")

# Check 1.3: Confidence scores validity
print("\n--- Check 1.3: Confidence Scores ---")
invalid_conf = ((df_aligned['confidence'] < 0) | (df_aligned['confidence'] > 1)).sum()
if invalid_conf > 0:
    add_issue('critical', f"Found {invalid_conf} confidence scores outside [0,1] range")
else:
    print("✅ All confidence scores in valid range [0,1]")

conf_stats = {
    'mean': df_aligned['confidence'].mean(),
    'median': df_aligned['confidence'].median(),
    'min': df_aligned['confidence'].min(),
    'max': df_aligned['confidence'].max(),
    'std': df_aligned['confidence'].std()
}
validation_results['statistics']['confidence'] = conf_stats
print(f"   Mean: {conf_stats['mean']:.3f}, Median: {conf_stats['median']:.3f}")
print(f"   Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")

if conf_stats['mean'] < 0.5:
    add_issue('warning', f"Low mean confidence score: {conf_stats['mean']:.3f}")

# Check 1.4: Segment coverage
print("\n--- Check 1.4: Segment Coverage ---")
segment_counts = df_aligned['segment_id'].value_counts().sort_index()
print(f"Total segments: {len(segment_counts)}")
for seg_id, count in segment_counts.items():
    print(f"   Segment {seg_id}: {count:,} rows ({count/len(df_aligned)*100:.1f}%)")
validation_results['statistics']['segments'] = segment_counts.to_dict()

# Check 1.5: Temporal coverage
print("\n--- Check 1.5: Temporal Coverage ---")
time_range = (df_aligned['aligned_time'].min(), df_aligned['aligned_time'].max())
duration_days = (time_range[1] - time_range[0]).total_seconds() / 86400
print(f"Time range: {time_range[0]} to {time_range[1]}")
print(f"Duration: {duration_days:.1f} days")
validation_results['statistics']['time_range'] = {
    'start': str(time_range[0]),
    'end': str(time_range[1]),
    'duration_days': duration_days
}

# Check 1.6: Duplicate timestamps in parquet (expected)
print("\n--- Check 1.6: Duplicate Timestamps in Parquet ---")
duplicates = df_aligned.duplicated('aligned_time', keep=False).sum()
print(f"Duplicate timestamps: {duplicates:,} ({duplicates/len(df_aligned)*100:.1f}%)")
print("ℹ️  Note: Duplicates in parquet are normal - they get spread during reconstruction")
validation_results['statistics']['parquet_duplicates'] = duplicates

# =============================================================================
# SECTION 2: VALIDATE RECONSTRUCTED FILES
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 2: RECONSTRUCTED FILES VALIDATION")
print("=" * 100)

# Check 2.1: File structure
print("\n--- Check 2.1: File Structure ---")
if not RECONSTRUCTED_DIR.exists():
    add_issue('critical', f"Reconstructed directory not found: {RECONSTRUCTED_DIR}")
    exit(1)

# Find all reconstructed files
all_files = list(RECONSTRUCTED_DIR.glob("**/*.TXT"))
print(f"Found {len(all_files)} .TXT files")
validation_results['statistics']['total_files'] = len(all_files)

if len(all_files) == 0:
    add_issue('critical', "No reconstructed files found!")
    exit(1)

# Check month folders
month_folders = [f.name for f in RECONSTRUCTED_DIR.iterdir() if f.is_dir() and f.name not in ['aligned_segments', 'validation', 'logs']]
print(f"Month folders: {sorted(month_folders)}")

# Check 2.2: Filename format
print("\n--- Check 2.2: Filename Format ---")
filename_pattern = re.compile(r'^\d{8}\.TXT$')
invalid_names = [f.name for f in all_files if not filename_pattern.match(f.name)]
if invalid_names:
    add_issue('warning', f"Found {len(invalid_names)} files with invalid names: {invalid_names[:5]}")
else:
    print("✅ All filenames follow YYMMDDHH.TXT format")

# =============================================================================
# SECTION 3: DEEP FILE CONTENT VALIDATION
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 3: DEEP FILE CONTENT VALIDATION")
print("=" * 100)
print("Analyzing file contents in detail... (this may take 1-2 minutes)")

file_issues = []
timestamp_stats = {
    'total_timestamps': 0,
    'invalid_timestamps': 0,
    'duplicate_timestamps_in_files': 0,
    'non_correct_intervals': 0, 
    'time_reversals_in_files': 0,
    'hour_mismatches': 0
}
lines_per_file = []
intervals_all = []
gap_sizes = []

print(f"\nChecking {len(all_files)} files...")

for file_idx, filepath in enumerate(all_files):
    if file_idx % 100 == 0:
        print(f"  Progress: {file_idx}/{len(all_files)} files checked...")
    
    filename = filepath.name
    
    try:
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        lines_per_file.append(len(lines))
        
        # Parse timestamps and data
        timestamps = []
        data_rows = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                        # Skip header line
            if line_num == 1 and 'Original_RTC' in line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                file_issues.append({
                    'file': filename,
                    'line': line_num,
                    'issue': 'Invalid line format',
                    'content': line[:50]
                })
                continue
            
            # Extract timestamp (last element)
            ts_str = parts[-1].strip()
            
            try:
                ts = datetime.strptime(ts_str, config.OUTPUT_FORMAT)
                timestamps.append(ts)
                data_rows.append(parts[:-1])
                timestamp_stats['total_timestamps'] += 1
            except ValueError:
                timestamp_stats['invalid_timestamps'] += 1
                file_issues.append({
                    'file': filename,
                    'line': line_num,
                    'issue': 'Invalid timestamp format',
                    'content': ts_str
                })
        
        if len(timestamps) == 0:
            file_issues.append({
                'file': filename,
                'issue': 'No valid timestamps found'
            })
            continue
        
        # Check 3.1: Duplicate timestamps within file
        if len(timestamps) != len(set(timestamps)):
            dup_count = len(timestamps) - len(set(timestamps))
            timestamp_stats['duplicate_timestamps_in_files'] += dup_count
            file_issues.append({
                'file': filename,
                'issue': f'Contains {dup_count} duplicate timestamps'
            })
        
        # Check 3.2: Chronological order
        sorted_ts = sorted(timestamps)
        if timestamps != sorted_ts:
            timestamp_stats['time_reversals_in_files'] += 1
            # Find reversals
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    file_issues.append({
                        'file': filename,
                        'line': i+1,
                        'issue': f'Time reversal: {timestamps[i-1]} -> {timestamps[i]}'
                    })
                    break
        
        # Check 3.3: Time intervals
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals_all.append(interval)
                
                # Check if interval is NOT correct (with tolerance)
                if abs(interval - EXPECTED_INTERVAL_SECONDS) > TOLERANCE_SECONDS:
                    timestamp_stats['non_correct_intervals'] += 1
                    
                    # Large gaps
                    if interval > MAX_ACCEPTABLE_GAP_MINUTES * 60:
                        gap_sizes.append(interval)
                        file_issues.append({
                            'file': filename,
                            'line': i+1,
                            'issue': f'Large gap: {interval/60:.1f} minutes',
                            'severity': 'info'
                        })
        
        # Check 3.4: Hour placement
        expected_hour = int(filename[6:8])  # Extract HH from YYMMDDHH
        actual_hours = [ts.hour for ts in timestamps]
        mismatches = sum(1 for h in actual_hours if h != expected_hour)
        
        if mismatches > 0:
            mismatch_pct = (mismatches / len(timestamps)) * 100
            if mismatch_pct > 20:  # More than 20% in wrong hour
                timestamp_stats['hour_mismatches'] += 1
                file_issues.append({
                    'file': filename,
                    'issue': f'{mismatches}/{len(timestamps)} timestamps in wrong hour (expected hour {expected_hour})',
                    'severity': 'warning'
                })
        
        # Check 3.5: Data values
        for row_num, row in enumerate(data_rows):
            if len(row) < 5:
                file_issues.append({
                    'file': filename,
                    'line': row_num + 1,
                    'issue': f'Insufficient data columns: {len(row)} (expected ≥5)',
                    'severity': 'warning'
                })
    
    except Exception as e:
        file_issues.append({
            'file': filename,
            'issue': f'Error reading file: {str(e)}',
            'severity': 'critical'
        })

print(f"✅ Completed analysis of {len(all_files)} files")

# =============================================================================
# SECTION 4: STATISTICAL ANALYSIS
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 4: STATISTICAL ANALYSIS")
print("=" * 100)

print("\n--- Check 4.1: File Size Distribution ---")
if lines_per_file:
    lines_array = np.array(lines_per_file)
    print(f"Mean lines per file: {lines_array.mean():.0f}")
    print(f"Median lines per file: {np.median(lines_array):.0f}")
    print(f"Min lines: {lines_array.min()}")
    print(f"Max lines: {lines_array.max()}")
    print(f"Std dev: {lines_array.std():.0f}")
    
    expected_full_hour = 3600 // EXPECTED_INTERVAL_SECONDS  # Device-specific
    coverage = (lines_array.mean() / expected_full_hour) * 100
    print(f"\nExpected lines per hour (@ {EXPECTED_INTERVAL_SECONDS}sec): {expected_full_hour}")  # ← FIXED
    print(f"Actual coverage: {coverage:.1f}%")
    
    validation_results['statistics']['file_sizes'] = {
        'mean': float(lines_array.mean()),
        'median': float(np.median(lines_array)),
        'min': int(lines_array.min()),
        'max': int(lines_array.max()),
        'coverage_pct': coverage
    }
    
    if coverage < 25:
        add_issue('warning', f"Very low data coverage: {coverage:.1f}% (device was off most of the time)")
    elif coverage < 50:
        add_issue('info', f"Moderate data coverage: {coverage:.1f}% (device had significant downtime)")

print("\n--- Check 4.2: Time Interval Analysis ---")
if intervals_all:
    intervals_array = np.array(intervals_all)
    
    print(f"Total intervals analyzed: {len(intervals_array):,}")
    print(f"Mean interval: {intervals_array.mean():.2f} seconds")
    print(f"Median interval: {np.median(intervals_array):.2f} seconds")
    print(f"Std dev: {intervals_array.std():.2f} seconds")
    
    # Categorize intervals (device-specific)
    exactly_correct = np.sum(np.abs(intervals_array - EXPECTED_INTERVAL_SECONDS) < TOLERANCE_SECONDS)
    near_correct = np.sum((intervals_array >= EXPECTED_INTERVAL_SECONDS*0.8) & 
                          (intervals_array <= EXPECTED_INTERVAL_SECONDS*1.2))
    large_gaps = np.sum(intervals_array > 60)
    
    print(f"\nExactly {EXPECTED_INTERVAL_SECONDS} seconds (±{TOLERANCE_SECONDS}s): {exactly_correct:,} ({exactly_correct/len(intervals_array)*100:.1f}%)")
    print(f"Near {EXPECTED_INTERVAL_SECONDS} seconds (±20%): {near_correct:,} ({near_correct/len(intervals_array)*100:.1f}%)")
    print(f"Large gaps (>60s): {large_gaps:,} ({large_gaps/len(intervals_array)*100:.1f}%)")
    
    validation_results['statistics']['intervals'] = {
        'mean': float(intervals_array.mean()),
        'median': float(np.median(intervals_array)),
        'exactly_correct_pct': float(exactly_correct/len(intervals_array)*100),
        'near_correct_pct': float(near_correct/len(intervals_array)*100),
        'large_gaps_count': int(large_gaps)
    }
    
    if exactly_correct / len(intervals_array) < 0.9:
        add_issue('warning', f"Only {exactly_correct/len(intervals_array)*100:.1f}% of intervals are exactly {EXPECTED_INTERVAL_SECONDS} seconds")

print("\n--- Check 4.3: Gap Analysis ---")
if gap_sizes:
    gaps_array = np.array(gap_sizes) / 60  # Convert to minutes
    print(f"Total large gaps (>{MAX_ACCEPTABLE_GAP_MINUTES} min): {len(gaps_array)}")
    print(f"Mean gap size: {gaps_array.mean():.1f} minutes")
    print(f"Median gap size: {np.median(gaps_array):.1f} minutes")
    print(f"Max gap: {gaps_array.max():.1f} minutes ({gaps_array.max()/60:.1f} hours)")
    print(f"Total time in gaps: {gaps_array.sum():.1f} minutes ({gaps_array.sum()/60:.1f} hours)")
    
    validation_results['statistics']['gaps'] = {
        'count': len(gaps_array),
        'mean_minutes': float(gaps_array.mean()),
        'max_minutes': float(gaps_array.max()),
        'total_hours': float(gaps_array.sum()/60)
    }
    
    if len(gaps_array) > 100:
        add_issue('info', f"Many gaps detected ({len(gaps_array)}) - indicates frequent device downtime")
else:
    print("✅ No large gaps detected")
    
# =============================================================================
# SECTION 5: CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 5: CROSS-VALIDATION (Parquet vs Files)")
print("=" * 100)

print("\n--- Check 5.1: Row Count Consistency ---")
total_file_lines = sum(lines_per_file)
parquet_rows = len(df_aligned)

print(f"Rows in parquet: {parquet_rows:,}")
print(f"Lines in files: {total_file_lines:,}")
print(f"Difference: {abs(parquet_rows - total_file_lines):,}")

if parquet_rows != total_file_lines:
    diff_pct = abs(parquet_rows - total_file_lines) / parquet_rows * 100
    if diff_pct > 1:
        add_issue('critical', f"Row count mismatch: {diff_pct:.1f}% difference between parquet and files")
    else:
        add_issue('info', f"Minor row count difference: {diff_pct:.3f}%")
else:
    print("✅ Perfect match!")

validation_results['statistics']['row_counts'] = {
    'parquet': parquet_rows,
    'files': total_file_lines,
    'match': parquet_rows == total_file_lines
}

# =============================================================================
# SECTION 6: QUALITY SCORES
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 6: QUALITY SCORES")
print("=" * 100)

# Calculate quality scores
quality_scores = {}

# Timestamp quality (0-100)
if timestamp_stats['total_timestamps'] > 0:
    valid_ts_pct = (1 - timestamp_stats['invalid_timestamps'] / timestamp_stats['total_timestamps']) * 100
    quality_scores['timestamp_validity'] = valid_ts_pct
    print(f"Timestamp Validity: {valid_ts_pct:.1f}%")

# Chronological quality
if timestamp_stats['total_timestamps'] > 0:
    chron_pct = (1 - timestamp_stats['time_reversals_in_files'] / len(all_files)) * 100
    quality_scores['chronological'] = chron_pct
    print(f"Chronological Order: {chron_pct:.1f}%")

# Interval consistency (device-specific)
if len(intervals_all) > 0:
    exact_interval_pct = (np.sum(np.abs(np.array(intervals_all) - EXPECTED_INTERVAL_SECONDS) < TOLERANCE_SECONDS) / len(intervals_all)) * 100
    quality_scores['interval_consistency'] = exact_interval_pct
    print(f"{EXPECTED_INTERVAL_SECONDS}-Second Interval Consistency: {exact_interval_pct:.1f}%")

# Duplicate-free score
if timestamp_stats['total_timestamps'] > 0:
    dup_free_pct = (1 - timestamp_stats['duplicate_timestamps_in_files'] / timestamp_stats['total_timestamps']) * 100
    quality_scores['duplicate_free'] = dup_free_pct
    print(f"Duplicate-Free: {dup_free_pct:.1f}%")

# Overall quality score (weighted average)
overall_quality = (
    quality_scores.get('timestamp_validity', 0) * 0.3 +
    quality_scores.get('chronological', 0) * 0.3 +
    quality_scores.get('interval_consistency', 0) * 0.2 +
    quality_scores.get('duplicate_free', 0) * 0.2
)
quality_scores['overall'] = overall_quality

print(f"\n{'='*50}")
print(f"OVERALL QUALITY SCORE: {overall_quality:.1f}/100")
print(f"{'='*50}")

if overall_quality >= 95:
    print("🏆 EXCELLENT - Publication quality data!")
elif overall_quality >= 85:
    print("✅ GOOD - Data is scientifically usable")
elif overall_quality >= 70:
    print("⚠️  FAIR - Data is usable but has some issues")
else:
    print("❌ POOR - Data has significant quality issues")

validation_results['quality_scores'] = quality_scores

# =============================================================================
# SECTION 7: FILE-LEVEL ISSUE REPORT
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 7: DETAILED ISSUE REPORT")
print("=" * 100)

# Categorize issues
critical_file_issues = [i for i in file_issues if i.get('severity') == 'critical']
warning_file_issues = [i for i in file_issues if i.get('severity') == 'warning']
info_file_issues = [i for i in file_issues if i.get('severity') == 'info']
other_file_issues = [i for i in file_issues if 'severity' not in i]

print(f"\nTotal file-level issues found: {len(file_issues)}")
print(f"  Critical: {len(critical_file_issues)}")
print(f"  Warnings: {len(warning_file_issues)}")
print(f"  Info: {len(info_file_issues)}")
print(f"  Other: {len(other_file_issues)}")

if critical_file_issues:
    print(f"\n❌ CRITICAL FILE ISSUES ({len(critical_file_issues)}):")
    for issue in critical_file_issues[:10]:
        print(f"   {issue}")
    if len(critical_file_issues) > 10:
        print(f"   ... and {len(critical_file_issues) - 10} more")

if warning_file_issues:
    print(f"\n⚠️  WARNING FILE ISSUES ({len(warning_file_issues)}):")
    for issue in warning_file_issues[:10]:
        print(f"   File: {issue['file']}, Issue: {issue['issue']}")
    if len(warning_file_issues) > 10:
        print(f"   ... and {len(warning_file_issues) - 10} more")

# =============================================================================
# SECTION 8: TIMESTAMP INTEGRITY
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 8: TIMESTAMP INTEGRITY SUMMARY")
print("=" * 100)

print(f"\nTotal timestamps validated: {timestamp_stats['total_timestamps']:,}")
print(f"Invalid timestamps: {timestamp_stats['invalid_timestamps']:,} ({timestamp_stats['invalid_timestamps']/max(timestamp_stats['total_timestamps'],1)*100:.3f}%)")
print(f"Duplicate timestamps in files: {timestamp_stats['duplicate_timestamps_in_files']:,}")
print(f"Non-{EXPECTED_INTERVAL_SECONDS}-second intervals: {timestamp_stats['non_correct_intervals']:,} ({timestamp_stats['non_correct_intervals']/max(len(intervals_all),1)*100:.1f}%)")
print(f"Files with time reversals: {timestamp_stats['time_reversals_in_files']}")
print(f"Files with hour mismatches: {timestamp_stats['hour_mismatches']}")

validation_results['timestamp_integrity'] = timestamp_stats

# Add to issues
if timestamp_stats['invalid_timestamps'] > 0:
    add_issue('critical', f"Found {timestamp_stats['invalid_timestamps']} invalid timestamps")

if timestamp_stats['duplicate_timestamps_in_files'] > 10:
    add_issue('critical', f"Found {timestamp_stats['duplicate_timestamps_in_files']} duplicate timestamps in output files")

if timestamp_stats['time_reversals_in_files'] > 0:
    add_issue('critical', f"Found time reversals in {timestamp_stats['time_reversals_in_files']} files")

# =============================================================================
# SECTION 9: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 9: FINAL VALIDATION SUMMARY")
print("=" * 100)

print(f"\n{'='*50}")
print("ISSUE SUMMARY")
print(f"{'='*50}")
print(f"Critical Issues: {len(validation_results['critical_issues'])}")
print(f"Warnings: {len(validation_results['warnings'])}")
print(f"Info Messages: {len(validation_results['info'])}")

if validation_results['critical_issues']:
    print(f"\n❌ CRITICAL ISSUES ({len(validation_results['critical_issues'])}):")
    for issue in validation_results['critical_issues']:
        print(f"   • {issue}")

if validation_results['warnings']:
    print(f"\n⚠️  WARNINGS ({len(validation_results['warnings'])}):")
    for warning in validation_results['warnings'][:10]:
        print(f"   • {warning}")
    if len(validation_results['warnings']) > 10:
        print(f"   ... and {len(validation_results['warnings']) - 10} more")

if validation_results['info']:
    print(f"\nℹ️  INFO ({len(validation_results['info'])}):")
    for info in validation_results['info'][:5]:
        print(f"   • {info}")
    if len(validation_results['info']) > 5:
        print(f"   ... and {len(validation_results['info']) - 5} more")

# =============================================================================
# SECTION 10: VERDICT
# =============================================================================
print("\n" + "=" * 100)
print("FINAL VERDICT")
print("=" * 100)

verdict_score = 0
verdict_details = []

# Scoring criteria
if len(validation_results['critical_issues']) == 0:
    verdict_score += 40
    verdict_details.append("✅ No critical issues")
else:
    verdict_details.append(f"❌ {len(validation_results['critical_issues'])} critical issues found")

if overall_quality >= 90:
    verdict_score += 30
    verdict_details.append("✅ Excellent quality score")
elif overall_quality >= 80:
    verdict_score += 20
    verdict_details.append("⚠️  Good quality score")
elif overall_quality >= 70:
    verdict_score += 10
    verdict_details.append("⚠️  Fair quality score")
else:
    verdict_details.append("❌ Poor quality score")

if timestamp_stats['time_reversals_in_files'] == 0:
    verdict_score += 15
    verdict_details.append("✅ No time reversals")
else:
    verdict_details.append(f"❌ {timestamp_stats['time_reversals_in_files']} files with time reversals")

if timestamp_stats['duplicate_timestamps_in_files'] < 10:
    verdict_score += 15
    verdict_details.append("✅ Minimal duplicate timestamps")
else:
    verdict_details.append(f"❌ {timestamp_stats['duplicate_timestamps_in_files']} duplicate timestamps")

print(f"\nVerdict Score: {verdict_score}/100")
print("\nDetails:")
for detail in verdict_details:
    print(f"  {detail}")

print("\n" + "=" * 50)
if verdict_score >= 90:
    print("🏆 VERDICT: EXCELLENT - DATA IS PUBLICATION READY!")
    print("   Your pipeline has produced high-quality, scientifically valid data.")
    print("   Confidence level: VERY HIGH ✅✅✅")
elif verdict_score >= 75:
    print("✅ VERDICT: GOOD - DATA IS SCIENTIFICALLY USABLE")
    print("   Your pipeline has produced usable data with minor issues.")
    print("   Confidence level: HIGH ✅✅")
elif verdict_score >= 50:
    print("⚠️  VERDICT: ACCEPTABLE - DATA IS USABLE WITH CAUTION")
    print("   Your pipeline has produced data with some issues.")
    print("   Confidence level: MODERATE ⚠️")
else:
    print("❌ VERDICT: POOR - DATA NEEDS REVIEW")
    print("   Your pipeline has produced data with significant issues.")
    print("   Confidence level: LOW ❌")
print("=" * 50)

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 100)
print("SAVING VALIDATION REPORT")
print("=" * 100)

report_file = config.VALIDATION_DIR / "ultimate_validation_report.json"
config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Convert to JSON-serializable format
validation_results['verdict_score'] = verdict_score
validation_results['verdict_details'] = verdict_details
validation_results['timestamp'] = datetime.now().isoformat()

with open(report_file, 'w') as f:
    json.dump(validation_results, f, indent=2, default=str)

print(f"✅ Full validation report saved: {report_file}")

# Also save a human-readable summary
summary_file = config.VALIDATION_DIR / "validation_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(f"ULTIMATE VALIDATION SUMMARY - Device {config.DEVICE_ID}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Validation Date: {datetime.now()}\n")
    f.write(f"Overall Quality Score: {overall_quality:.1f}/100\n")
    f.write(f"Verdict Score: {verdict_score}/100\n\n")
    
    f.write("CRITICAL ISSUES:\n")
    if validation_results['critical_issues']:
        for issue in validation_results['critical_issues']:
            f.write(f"  • {issue}\n")
    else:
        f.write("  None ✅\n")
    
    f.write("\nWARNINGS:\n")
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            f.write(f"  • {warning}\n")
    else:
        f.write("  None ✅\n")
    
    f.write(f"\nKEY STATISTICS:\n")
    f.write(f"  Total rows processed: {parquet_rows:,}\n")
    f.write(f"  Total files created: {len(all_files)}\n")
    f.write(f"  Mean confidence: {conf_stats['mean']:.3f}\n")
    if 'file_sizes' in validation_results['statistics']:
        f.write(f"  Data coverage: {validation_results['statistics']['file_sizes']['coverage_pct']:.1f}%\n")
    f.write(f"  Chronological files: {chron_pct:.1f}%\n")

print(f"✅ Human-readable summary saved: {summary_file}")

print("\n" + "=" * 100)
print("ULTIMATE VALIDATION COMPLETE!")
print("=" * 100)
print("\n✅ Validation complete! Review the report above.")
print(f"📄 Detailed report: {report_file}")
print(f"📄 Summary: {summary_file}\n")