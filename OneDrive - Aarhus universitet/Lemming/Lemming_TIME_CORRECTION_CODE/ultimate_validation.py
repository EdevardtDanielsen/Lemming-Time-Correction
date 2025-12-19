# =============================================================================
# ULTIMATE / TRUE VALIDATION SCRIPT
# =============================================================================
# Comprehensive deep-dive validation of the entire RTC-TTN alignment pipeline.
# Works for any device (H1, C3, A1, B1, etc.), using paths from config.py.
#
# Why this exists: After the pipeline runs, we need to verify output quality.
# This script checks for: chronological order, duplicate timestamps, gaps,
# interval consistency, and cross-validates parquet vs file outputs.
# Also imports and reports on the pipeline's own direct + cross-validation.
#
# Author: Edevardt Johan Danielsen (ED)
# Last edited: 21/11/2025
# =============================================================================

import sys
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import config to get device-specific paths and formats
import config

# =============================================================================
# BASIC SETUP
# =============================================================================
print("=" * 100)
print("ULTIMATE PIPELINE VALIDATION - COMPREHENSIVE QUALITY CHECK")
print("=" * 100)
print(f"Device: {config.DEVICE_ID}")
print("\nThis will perform an exhaustive check of all pipeline outputs.\n")

BASE_DIR = config.OUTPUT_DIR
RECONSTRUCTED_DIR = config.RECONSTRUCTED_DATA_DIR
ALIGNED_DIR = config.ALIGNED_DATA_DIR
VALIDATION_DIR = config.VALIDATION_DIR

# Expected parameters
EXPECTED_INTERVAL_SECONDS = 2
TOLERANCE_SECONDS = 0.1          # Allow small floating point differences
MAX_ACCEPTABLE_GAP_MINUTES = 10  # Flag gaps larger than this

# Results tracking
validation_results = {
    "device_id": config.DEVICE_ID,
    "critical_issues": [],
    "warnings": [],
    "info": [],
    "statistics": {},
}

# Register an issue and print it to the console with severity level
def add_issue(level: str, message: str):
    if level == "critical":
        validation_results["critical_issues"].append(message)
        print(f"[CRITICAL] {message}")
    elif level == "warning":
        validation_results["warnings"].append(message)
        print(f"[WARNING] {message}")
    else:
        validation_results["info"].append(message)
        print(f"[INFO] {message}")

# =============================================================================
# SECTION 0: IMPORT PIPELINE VALIDATION SUMMARY (DIRECT + CROSS-VAL)
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 0: PIPELINE VALIDATION SUMMARY (DIRECT + CROSS-VAL)")
print("=" * 100)

summary_path = VALIDATION_DIR / "validation_summary.json"
pipeline_validation = None

if summary_path.exists():
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            pipeline_validation = json.load(f)
        validation_results["statistics"]["pipeline_validation"] = pipeline_validation

        direct = pipeline_validation.get("direct_validation", {})
        cross = pipeline_validation.get("cross_validation", {})

        print("\nDirect validation:")
        print(f"  N points: {direct.get('n_points')}")
        print(f"  MAE: {direct.get('mae_seconds')} s")
        print(f"  95%: ±{direct.get('p95_seconds')} s")

        print("\nCross-validation:")
        print(f"  N points: {cross.get('n_points')}")
        print(f"  MAE: {cross.get('mae_seconds')} s")
        print(f"  95%: ±{cross.get('p95_seconds')} s")

        mae_cross = cross.get("mae_seconds", None)
        if mae_cross is not None and mae_cross > 2 * 3600:
            add_issue("warning", f"High cross-validation MAE: {mae_cross:.1f} s (> 2 hours)")
    except Exception as e:
        add_issue("warning", f"Could not read validation_summary.json: {e}")
else:
    add_issue("warning", f"No validation_summary.json found at {summary_path}")

# =============================================================================
# SECTION 1: VALIDATE ALIGNED PARQUET FILE
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 1: ALIGNED DATA (PARQUET) VALIDATION")
print("=" * 100)

aligned_file = ALIGNED_DIR / "aligned_data.parquet"
print(f"\nLoading aligned data from: {aligned_file}")

if not aligned_file.exists():
    add_issue("critical", f"Aligned data file not found: {aligned_file}")
    print("\nCannot proceed without aligned data file.")
    sys.exit(1)

try:
    df_aligned = pd.read_parquet(aligned_file)
    n_rows_aligned = len(df_aligned)
    print(f"Loaded {n_rows_aligned:,} aligned rows.")
    validation_results["statistics"]["total_aligned_rows"] = n_rows_aligned
except Exception as e:
    add_issue("critical", f"Cannot load aligned data: {e}")
    sys.exit(1)

# --- Check 1.1: Required columns ---
print("\n--- Check 1.1: Required Columns ---")
required_cols = ["aligned_time", "original_rtc", "confidence", "segment_id", "data_values"]
missing_cols = [c for c in required_cols if c not in df_aligned.columns]
if missing_cols:
    add_issue("critical", f"Missing required columns in aligned parquet: {missing_cols}")
else:
    print("All required columns present.")

# --- Check 1.2: Data types ---
print("\n--- Check 1.2: Data Types ---")
if not pd.api.types.is_datetime64_any_dtype(df_aligned["aligned_time"]):
    add_issue("critical", "aligned_time is not datetime type")
if not pd.api.types.is_datetime64_any_dtype(df_aligned["original_rtc"]):
    add_issue("critical", "original_rtc is not datetime type")
if not (
    pd.api.types.is_float_dtype(df_aligned["confidence"])
    or pd.api.types.is_integer_dtype(df_aligned["confidence"])
):
    add_issue("warning", "confidence is not numeric type")
print("Data types checked.")

# --- Check 1.3: Confidence scores ---
print("\n--- Check 1.3: Confidence Scores ---")
invalid_conf = ((df_aligned["confidence"] < 0) | (df_aligned["confidence"] > 1)).sum()
if invalid_conf > 0:
    add_issue("critical", f"Found {invalid_conf} confidence scores outside [0,1].")
else:
    print("All confidence scores in [0, 1].")

conf_stats = {
    "mean": float(df_aligned["confidence"].mean()),
    "median": float(df_aligned["confidence"].median()),
    "min": float(df_aligned["confidence"].min()),
    "max": float(df_aligned["confidence"].max()),
    "std": float(df_aligned["confidence"].std()),
}
validation_results["statistics"]["confidence"] = conf_stats

print(
    f"Confidence - mean: {conf_stats['mean']:.3f}, "
    f"median: {conf_stats['median']:.3f}, "
    f"range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]"
)
if conf_stats["mean"] < 0.5:
    add_issue("warning", f"Low mean confidence score: {conf_stats['mean']:.3f}")

# --- Check 1.4: Segment coverage ---
print("\n--- Check 1.4: Segment Coverage ---")
segment_counts = df_aligned["segment_id"].value_counts().sort_index()
print(f"Total segments: {len(segment_counts)}")
for seg_id, count in segment_counts.items():
    pct = count / n_rows_aligned * 100
    print(f"  Segment {seg_id}: {count:,} rows ({pct:.1f}%)")
validation_results["statistics"]["segments"] = segment_counts.to_dict()

# --- Check 1.5: Temporal coverage ---
print("\n--- Check 1.5: Temporal Coverage ---")
t_min = df_aligned["aligned_time"].min()
t_max = df_aligned["aligned_time"].max()
duration_days = (t_max - t_min).total_seconds() / 86400.0
print(f"Aligned time range: {t_min} to {t_max} ({duration_days:.1f} days)")
validation_results["statistics"]["time_range"] = {
    "start": str(t_min),
    "end": str(t_max),
    "duration_days": float(duration_days),
}

# --- Check 1.6: Duplicate timestamps (in parquet) ---
print("\n--- Check 1.6: Duplicate aligned timestamps in parquet ---")
duplicates = df_aligned.duplicated("aligned_time", keep=False).sum()
dup_pct = duplicates / max(n_rows_aligned, 1) * 100
print(f"Duplicate aligned_time rows: {duplicates:,} ({dup_pct:.1f}%)")
print("Note: some duplicates in parquet can be expected; they are spread across files later.")
validation_results["statistics"]["parquet_duplicates"] = int(duplicates)

# =============================================================================
# SECTION 2: VALIDATE RECONSTRUCTED FILES
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 2: RECONSTRUCTED FILES VALIDATION")
print("=" * 100)

print(f"Reconstructed directory: {RECONSTRUCTED_DIR}")

if not RECONSTRUCTED_DIR.exists():
    add_issue("critical", f"Reconstructed directory not found: {RECONSTRUCTED_DIR}")
    sys.exit(1)

# Find all reconstructed TXT files (recursively)
all_files = list(RECONSTRUCTED_DIR.glob("**/*.TXT"))
n_files = len(all_files)
print(f"Found {n_files} reconstructed .TXT files.")
validation_results["statistics"]["total_files"] = n_files

if n_files == 0:
    add_issue("critical", "No reconstructed TXT files found.")
    sys.exit(1)

# Month folders (excluding special subdirs)
month_folders = [
    f.name
    for f in RECONSTRUCTED_DIR.iterdir()
    if f.is_dir() and f.name not in ["aligned_segments", "validation", "logs"]
]
print(f"Month folders detected: {sorted(month_folders)}")

# --- Check 2.2: Filename format: YYMMDDHH.TXT ---
print("\n--- Check 2.2: Filename Format (YYMMDDHH.TXT) ---")
filename_pattern = re.compile(r"^\d{8}\.TXT$")
invalid_names = [f.name for f in all_files if not filename_pattern.match(f.name)]
if invalid_names:
    add_issue(
        "warning",
        f"Found {len(invalid_names)} files with unexpected names (showing up to 5): "
        f"{invalid_names[:5]}",
    )
else:
    print("All filenames follow YYMMDDHH.TXT format.")

# =============================================================================
# SECTION 3: DEEP FILE CONTENT VALIDATION
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 3: DEEP FILE CONTENT VALIDATION")
print("=" * 100)
print("Analyzing file contents in detail... this may take some time.\n")

file_issues = []
timestamp_stats = {
    "total_timestamps": 0,
    "invalid_timestamps": 0,
    "duplicate_timestamps_in_files": 0,
    "non_2sec_intervals": 0,
    "time_reversals_in_files": 0,
    "hour_mismatches": 0,
}

lines_per_file = []
intervals_all = []
gap_sizes = []
file_spans = []  # new: span info per file (for coverage analysis)

for idx, filepath in enumerate(all_files):
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{n_files} files checked...")

    filename = filepath.name

    try:
        # Read lines
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        lines_per_file.append(len(lines))

        timestamps = []
        data_rows = []

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 2:
                file_issues.append(
                    {
                        "file": filename,
                        "line": line_num,
                        "issue": "Invalid line format (less than 2 columns).",
                        "content": line[:80],
                        "severity": "warning",
                    }
                )
                continue

            ts_str = parts[-1].strip()
            try:
                ts = datetime.strptime(ts_str, config.OUTPUT_FORMAT)
                timestamps.append(ts)
                data_rows.append(parts[:-1])
                timestamp_stats["total_timestamps"] += 1
            except ValueError:
                timestamp_stats["invalid_timestamps"] += 1
                file_issues.append(
                    {
                        "file": filename,
                        "line": line_num,
                        "issue": "Invalid timestamp format.",
                        "content": ts_str,
                        "severity": "critical",
                    }
                )

        if len(timestamps) == 0:
            file_issues.append(
                {"file": filename, "issue": "No valid timestamps found in file.", "severity": "critical"}
            )
            continue

        # Sort timestamps to inspect span and intervals
        timestamps_sorted = sorted(timestamps)
        start_ts = timestamps_sorted[0]
        end_ts = timestamps_sorted[-1]
        duration_sec = (end_ts - start_ts).total_seconds()

        # Store per-file span
        file_spans.append(
            {
                "file": filename,
                "start": start_ts,
                "end": end_ts,
                "duration_sec": duration_sec,
            }
        )

        # --- Check 3.1: Duplicate timestamps within file ---
        if len(timestamps) != len(set(timestamps)):
            dup_count = len(timestamps) - len(set(timestamps))
            timestamp_stats["duplicate_timestamps_in_files"] += dup_count
            file_issues.append(
                {
                    "file": filename,
                    "issue": f"Contains {dup_count} duplicate timestamps.",
                    "severity": "warning",
                }
            )

        # --- Check 3.2: Chronological order within file ---
        # (We compare unsorted timestamps to sorted order)
        if timestamps != timestamps_sorted:
            timestamp_stats["time_reversals_in_files"] += 1
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i - 1]:
                    file_issues.append(
                        {
                            "file": filename,
                            "line": i + 1,
                            "issue": f"Time reversal: {timestamps[i-1]} -> {timestamps[i]}",
                            "severity": "critical",
                        }
                    )
                    break

        # --- Check 3.3: Time intervals ---
        if len(timestamps_sorted) > 1:
            for i in range(1, len(timestamps_sorted)):
                interval = (timestamps_sorted[i] - timestamps_sorted[i - 1]).total_seconds()
                intervals_all.append(interval)

                # Count non-2sec intervals
                if abs(interval - EXPECTED_INTERVAL_SECONDS) > TOLERANCE_SECONDS:
                    timestamp_stats["non_2sec_intervals"] += 1

                    # Large gaps
                    if interval > MAX_ACCEPTABLE_GAP_MINUTES * 60:
                        gap_sizes.append(interval)
                        file_issues.append(
                            {
                                "file": filename,
                                "line": i + 1,
                                "issue": f"Large gap: {interval/60:.1f} minutes",
                                "severity": "info",
                            }
                        )

        # --- Check 3.4: Hour placement ---
        # Filename YYMMDDHH.TXT, hour is last two digits before extension
        try:
            expected_hour = int(filename[6:8])
            actual_hours = [ts.hour for ts in timestamps_sorted]
            mismatches = sum(1 for h in actual_hours if h != expected_hour)
            if mismatches > 0:
                mismatch_pct = mismatches / len(actual_hours) * 100
                if mismatch_pct > 20:  # more than 20 % of lines in wrong hour
                    timestamp_stats["hour_mismatches"] += 1
                    file_issues.append(
                        {
                            "file": filename,
                            "issue": f"{mismatches}/{len(actual_hours)} timestamps in wrong hour "
                                     f"(expected hour {expected_hour}).",
                            "severity": "warning",
                        }
                    )
        except ValueError:
            file_issues.append(
                {
                    "file": filename,
                    "issue": "Could not parse expected hour from filename.",
                    "severity": "warning",
                }
            )

        # --- Check 3.5: Data value columns ---
        for row_num, row in enumerate(data_rows, start=1):
            if len(row) < 5:
                file_issues.append(
                    {
                        "file": filename,
                        "line": row_num,
                        "issue": f"Insufficient data columns: {len(row)} (expected ≥5).",
                        "severity": "warning",
                    }
                )

    except Exception as e:
        file_issues.append(
            {
                "file": filename,
                "issue": f"Error reading file: {e}",
                "severity": "critical",
            }
        )

print(f"\nCompleted analysis of {n_files} files.")

# =============================================================================
# SECTION 4: STATISTICAL ANALYSIS
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 4: STATISTICAL ANALYSIS")
print("=" * 100)

# --- Check 4.1: File size distribution ---
print("\n--- Check 4.1: File Size (lines per file) ---")
if lines_per_file:
    lines_array = np.array(lines_per_file)
    mean_lines = lines_array.mean()
    median_lines = float(np.median(lines_array))
    min_lines = int(lines_array.min())
    max_lines = int(lines_array.max())
    std_lines = float(lines_array.std())

    print(f"Mean lines per file:   {mean_lines:.0f}")
    print(f"Median lines per file: {median_lines:.0f}")
    print(f"Min lines per file:    {min_lines}")
    print(f"Max lines per file:    {max_lines}")
    print(f"Std dev (lines/file):  {std_lines:.0f}")

    expected_full_hour = 3600 / EXPECTED_INTERVAL_SECONDS  # e.g. 1800 for 2 s interval
    coverage_pct = (mean_lines / expected_full_hour) * 100
    print(f"\nExpected lines per hour (@ {EXPECTED_INTERVAL_SECONDS}s): {expected_full_hour:.0f}")
    print(f"Actual mean coverage: {coverage_pct:.1f}%")

    validation_results["statistics"]["file_sizes"] = {
        "mean": float(mean_lines),
        "median": median_lines,
        "min": min_lines,
        "max": max_lines,
        "coverage_pct": float(coverage_pct),
    }

    if coverage_pct < 25:
        add_issue("warning", f"Very low average coverage: {coverage_pct:.1f}% (device often off).")
    elif coverage_pct < 50:
        add_issue("info", f"Moderate coverage: {coverage_pct:.1f}% (significant downtime).")

# --- Check 4.1b: Per-file time span & start time ---
print("\n--- Check 4.1b: Per-File Time Span and Start Offset ---")
if file_spans:
    durations = np.array([fs["duration_sec"] for fs in file_spans])
    starts_minutes = np.array(
        [fs["start"].minute + fs["start"].second / 60.0 for fs in file_spans]
    )

    print(f"Mean coverage per file:   {durations.mean() / 60:.1f} minutes")
    print(f"Median coverage per file: {np.median(durations) / 60:.1f} minutes")
    print(f"Min coverage:             {durations.min() / 60:.1f} minutes")
    print(f"Max coverage:             {durations.max() / 60:.1f} minutes")

    print(f"\nMean start offset from hour:   {starts_minutes.mean():.1f} minutes")
    print(f"Median start offset from hour: {np.median(starts_minutes):.1f} minutes")

    # Save statistics
    validation_results["statistics"]["file_spans"] = {
        "mean_minutes": float(durations.mean() / 60),
        "median_minutes": float(np.median(durations) / 60),
        "min_minutes": float(durations.min() / 60),
        "max_minutes": float(durations.max() / 60),
        "mean_start_minute": float(starts_minutes.mean()),
        "median_start_minute": float(np.median(starts_minutes)),
    }

    # Thresholds to flag "short" hours
    short_45 = [fs for fs in file_spans if fs["duration_sec"] < 45 * 60]
    short_30 = [fs for fs in file_spans if fs["duration_sec"] < 30 * 60]

    if short_45:
        add_issue(
            "warning",
            f"{len(short_45)} files have <45 minutes of coverage (out of {n_files}).",
        )

    if short_30:
        add_issue(
            "warning",
            f"{len(short_30)} files have <30 minutes of coverage (out of {n_files}).",
        )

    # Show a few examples of the shortest hours
    short_sorted = sorted(file_spans, key=lambda fs: fs["duration_sec"])[:10]
    print("\nExamples of shortest coverage files (up to 10):")
    for fs in short_sorted:
        mins = fs["duration_sec"] / 60.0
        print(
            f"  {fs['file']}: {mins:5.1f} min, "
            f"start={fs['start']}, end={fs['end']}"
        )
else:
    print("No file span information collected.")

# --- Check 4.2: Time Interval Analysis ---
print("\n--- Check 4.2: Time Interval Analysis ---")
if intervals_all:
    intervals_array = np.array(intervals_all)

    print(f"Total intervals analyzed: {len(intervals_array):,}")
    print(f"Mean interval:   {intervals_array.mean():.3f} s")
    print(f"Median interval: {np.median(intervals_array):.3f} s")
    print(f"Std dev:         {intervals_array.std():.3f} s")

    exactly_2sec = np.sum(np.abs(intervals_array - EXPECTED_INTERVAL_SECONDS) < TOLERANCE_SECONDS)
    near_2sec = np.sum((intervals_array >= 1.5) & (intervals_array <= 2.5))
    large_gaps = np.sum(intervals_array > 60)

    print(
        f"\nExactly {EXPECTED_INTERVAL_SECONDS}s (±{TOLERANCE_SECONDS}s): "
        f"{exactly_2sec:,} ({exactly_2sec / len(intervals_array) * 100:.1f}%)"
    )
    print(
        f"Near 2 s (1.5–2.5s): {near_2sec:,} "
        f"({near_2sec / len(intervals_array) * 100:.1f}%)"
    )
    print(
        f"Large gaps (>60s):   {large_gaps:,} "
        f"({large_gaps / len(intervals_array) * 100:.1f}%)"
    )

    validation_results["statistics"]["intervals"] = {
        "mean": float(intervals_array.mean()),
        "median": float(np.median(intervals_array)),
        "exactly_2sec_pct": float(exactly_2sec / len(intervals_array) * 100),
        "near_2sec_pct": float(near_2sec / len(intervals_array) * 100),
        "large_gaps_count": int(large_gaps),
    }

    if exactly_2sec / len(intervals_array) < 0.9:
        add_issue(
            "warning",
            f"Only {exactly_2sec / len(intervals_array) * 100:.1f}% of intervals are exactly "
            f"{EXPECTED_INTERVAL_SECONDS}s.",
        )
else:
    print("No intervals to analyze (not enough timestamps).")

# --- Check 4.3: Gap Analysis ---
print("\n--- Check 4.3: Gap Analysis (> 10 min) ---")
if gap_sizes:
    gaps_array = np.array(gap_sizes) / 60.0  # minutes
    print(f"Total large gaps: {len(gaps_array)}")
    print(f"Mean gap:   {gaps_array.mean():.1f} min")
    print(f"Median gap: {np.median(gaps_array):.1f} min")
    print(f"Max gap:    {gaps_array.max():.1f} min ({gaps_array.max()/60:.1f} h)")
    print(f"Total time in gaps: {gaps_array.sum():.1f} min ({gaps_array.sum()/60:.1f} h)")

    validation_results["statistics"]["gaps"] = {
        "count": int(len(gaps_array)),
        "mean_minutes": float(gaps_array.mean()),
        "max_minutes": float(gaps_array.max()),
        "total_hours": float(gaps_array.sum() / 60.0),
    }
else:
    print("No gaps > 10 minutes detected.")

# =============================================================================
# SECTION 5: CROSS-VALIDATION (PARQUET VS FILES)
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 5: CROSS-VALIDATION (Parquet vs Reconstructed Files)")
print("=" * 100)

total_file_lines = sum(lines_per_file)
parquet_rows = n_rows_aligned

print(f"Rows in aligned parquet: {parquet_rows:,}")
print(f"Total lines in TXT files: {total_file_lines:,}")
diff = abs(parquet_rows - total_file_lines)
print(f"Difference: {diff:,}")

if parquet_rows != total_file_lines:
    diff_pct = diff / max(parquet_rows, 1) * 100.0
    if diff_pct > 1.0:
        add_issue(
            "critical",
            f"Row count mismatch between parquet and TXT files: {diff_pct:.1f}% difference.",
        )
    else:
        add_issue(
            "info",
            f"Minor row count mismatch between parquet and TXT files: {diff_pct:.3f}%.",
        )
else:
    print("Perfect row count match between parquet and TXT files.")

validation_results["statistics"]["row_counts"] = {
    "parquet": parquet_rows,
    "files": total_file_lines,
    "match": parquet_rows == total_file_lines,
}

# =============================================================================
# SECTION 6: QUALITY SCORES
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 6: QUALITY SCORES")
print("=" * 100)

quality_scores = {}

# Timestamp validity
if timestamp_stats["total_timestamps"] > 0:
    valid_ts_pct = (
        1 - timestamp_stats["invalid_timestamps"] / timestamp_stats["total_timestamps"]
    ) * 100.0
    quality_scores["timestamp_validity"] = float(valid_ts_pct)
    print(f"Timestamp Validity: {valid_ts_pct:.1f}%")

# Chronological quality (per-file)
if n_files > 0:
    chron_pct = (
        1 - timestamp_stats["time_reversals_in_files"] / n_files
    ) * 100.0
    quality_scores["chronological"] = float(chron_pct)
    print(f"Chronological Order (files without reversals): {chron_pct:.1f}%")
else:
    chron_pct = 0.0

# Interval consistency
if intervals_all:
    intervals_array = np.array(intervals_all)
    exact_2sec_pct = (
        np.sum(np.abs(intervals_array - EXPECTED_INTERVAL_SECONDS) < TOLERANCE_SECONDS)
        / len(intervals_array)
        * 100.0
    )
    quality_scores["interval_consistency"] = float(exact_2sec_pct)
    print(f"2-second Interval Consistency: {exact_2sec_pct:.1f}%")
else:
    quality_scores["interval_consistency"] = 0.0
    exact_2sec_pct = 0.0

# Duplicate-free score
if timestamp_stats["total_timestamps"] > 0:
    dup_free_pct = (
        1
        - timestamp_stats["duplicate_timestamps_in_files"]
        / timestamp_stats["total_timestamps"]
    ) * 100.0
    quality_scores["duplicate_free"] = float(dup_free_pct)
    print(f"Duplicate-Free Timestamps: {dup_free_pct:.1f}%")
else:
    quality_scores["duplicate_free"] = 0.0

# Overall quality score (weighted)
overall_quality = (
    quality_scores.get("timestamp_validity", 0.0) * 0.3
    + quality_scores.get("chronological", 0.0) * 0.3
    + quality_scores.get("interval_consistency", 0.0) * 0.2
    + quality_scores.get("duplicate_free", 0.0) * 0.2
)
quality_scores["overall"] = float(overall_quality)

print("\n" + "-" * 50)
print(f"OVERALL QUALITY SCORE: {overall_quality:.1f} / 100")
print("-" * 50)

if overall_quality >= 95:
    print("Verdict: EXCELLENT - publication quality data.")
elif overall_quality >= 85:
    print("Verdict: GOOD - scientifically usable with minor issues.")
elif overall_quality >= 70:
    print("Verdict: FAIR - usable but with notable issues.")
else:
    print("Verdict: POOR - data needs careful review.")

validation_results["quality_scores"] = quality_scores

# =============================================================================
# SECTION 7: FILE-LEVEL ISSUE REPORT
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 7: DETAILED ISSUE REPORT")
print("=" * 100)

critical_file_issues = [i for i in file_issues if i.get("severity") == "critical"]
warning_file_issues = [i for i in file_issues if i.get("severity") == "warning"]
info_file_issues = [i for i in file_issues if i.get("severity") == "info"]
other_file_issues = [i for i in file_issues if "severity" not in i]

print(f"\nTotal file-level issues: {len(file_issues)}")
print(f"  Critical: {len(critical_file_issues)}")
print(f"  Warnings: {len(warning_file_issues)}")
print(f"  Info:     {len(info_file_issues)}")
print(f"  Other:    {len(other_file_issues)}")

if critical_file_issues:
    print("\nCritical file issues (showing up to 10):")
    for issue in critical_file_issues[:10]:
        print(f"  {issue}")
    if len(critical_file_issues) > 10:
        print(f"  ... and {len(critical_file_issues) - 10} more")

if warning_file_issues:
    print("\nWarning file issues (showing up to 10):")
    for issue in warning_file_issues[:10]:
        print(f"  File: {issue['file']}, Issue: {issue['issue']}")
    if len(warning_file_issues) > 10:
        print(f"  ... and {len(warning_file_issues) - 10} more")

# =============================================================================
# SECTION 8: TIMESTAMP INTEGRITY SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 8: TIMESTAMP INTEGRITY SUMMARY")
print("=" * 100)

total_ts = timestamp_stats["total_timestamps"]
invalid_ts = timestamp_stats["invalid_timestamps"]
dup_ts = timestamp_stats["duplicate_timestamps_in_files"]
non_2sec = timestamp_stats["non_2sec_intervals"]

print(f"\nTotal timestamps validated:      {total_ts:,}")
print(f"Invalid timestamps:              {invalid_ts:,}")
print(
    f"Invalid timestamp fraction:      "
    f"{invalid_ts / max(total_ts, 1) * 100:.3f}%"
)
print(f"Duplicate timestamps in files:   {dup_ts:,}")
print(
    f"Non-{EXPECTED_INTERVAL_SECONDS}s intervals: {non_2sec:,} "
    f"({non_2sec / max(len(intervals_all), 1) * 100:.1f}%)"
)
print(f"Files with time reversals:       {timestamp_stats['time_reversals_in_files']}")
print(f"Files with hour mismatches:      {timestamp_stats['hour_mismatches']}")

validation_results["timestamp_integrity"] = timestamp_stats

if invalid_ts > 0:
    add_issue("critical", f"Found {invalid_ts} invalid timestamps in TXT files.")
if dup_ts > 10:
    add_issue("critical", f"Found {dup_ts} duplicate timestamps in TXT files.")
if timestamp_stats["time_reversals_in_files"] > 0:
    add_issue(
        "critical",
        f"Found time reversals in {timestamp_stats['time_reversals_in_files']} files.",
    )

# =============================================================================
# SECTION 9: FINAL VALIDATION SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 9: FINAL VALIDATION SUMMARY")
print("=" * 100)

print("\n" + "-" * 50)
print("ISSUE SUMMARY")
print("-" * 50)
print(f"Critical Issues: {len(validation_results['critical_issues'])}")
print(f"Warnings:        {len(validation_results['warnings'])}")
print(f"Info Messages:   {len(validation_results['info'])}")

if validation_results["critical_issues"]:
    print("\nCritical issues:")
    for issue in validation_results["critical_issues"]:
        print(f"  - {issue}")

if validation_results["warnings"]:
    print("\nWarnings (showing up to 10):")
    for w in validation_results["warnings"][:10]:
        print(f"  - {w}")
    if len(validation_results["warnings"]) > 10:
        print(f"  ... and {len(validation_results['warnings']) - 10} more")

if validation_results["info"]:
    print("\nInfo (showing up to 5):")
    for info in validation_results["info"][:5]:
        print(f"  - {info}")
    if len(validation_results["info"]) > 5:
        print(f"  ... and {len(validation_results['info']) - 5} more")

# =============================================================================
# SECTION 10: VERDICT
# =============================================================================
print("\n" + "=" * 100)
print("FINAL VERDICT")
print("=" * 100)

verdict_score = 0
verdict_details = []

# 1) Critical issues
if len(validation_results["critical_issues"]) == 0:
    verdict_score += 40
    verdict_details.append("No critical issues.")
else:
    verdict_details.append(
        f"{len(validation_results['critical_issues'])} critical issues found."
    )

# 2) Overall quality
if overall_quality >= 90:
    verdict_score += 30
    verdict_details.append("Excellent overall quality score.")
elif overall_quality >= 80:
    verdict_score += 20
    verdict_details.append("Good overall quality score.")
elif overall_quality >= 70:
    verdict_score += 10
    verdict_details.append("Fair overall quality score.")
else:
    verdict_details.append("Poor overall quality score.")

# 3) Time reversals
if timestamp_stats["time_reversals_in_files"] == 0:
    verdict_score += 15
    verdict_details.append("No time reversals detected.")
else:
    verdict_details.append(
        f"{timestamp_stats['time_reversals_in_files']} files with time reversals."
    )

# 4) Duplicate timestamps
if timestamp_stats["duplicate_timestamps_in_files"] < 10:
    verdict_score += 15
    verdict_details.append("Minimal duplicate timestamps.")
else:
    verdict_details.append(
        f"{timestamp_stats['duplicate_timestamps_in_files']} duplicate timestamps."
    )

print(f"\nVerdict Score: {verdict_score} / 100")
print("Details:")
for d in verdict_details:
    print(f"  - {d}")

print("\n" + "-" * 50)
if verdict_score >= 90:
    print("VERDICT: EXCELLENT - DATA IS PUBLICATION READY.")
elif verdict_score >= 75:
    print("VERDICT: GOOD - DATA IS SCIENTIFICALLY USABLE.")
elif verdict_score >= 50:
    print("VERDICT: ACCEPTABLE - USE WITH CAUTION.")
else:
    print("VERDICT: POOR - DATA NEEDS REVIEW.")
print("-" * 50)

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 100)
print("SAVING VALIDATION REPORT")
print("=" * 100)

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

report_file = VALIDATION_DIR / "ultimate_validation_report.json"
summary_file = VALIDATION_DIR / "validation_summary_true.txt"

validation_results["verdict_score"] = verdict_score
validation_results["verdict_details"] = verdict_details
validation_results["timestamp"] = datetime.now().isoformat()

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(validation_results, f, indent=2, default=str)

print(f"Full validation report saved to: {report_file}")

with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"ULTIMATE VALIDATION SUMMARY - Device {config.DEVICE_ID}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Validation Date: {datetime.now()}\n")
    f.write(f"Overall Quality Score: {overall_quality:.1f} / 100\n")
    f.write(f"Verdict Score: {verdict_score} / 100\n\n")

    f.write("CRITICAL ISSUES:\n")
    if validation_results["critical_issues"]:
        for issue in validation_results["critical_issues"]:
            f.write(f"  - {issue}\n")
    else:
        f.write("  None\n")

    f.write("\nWARNINGS:\n")
    if validation_results["warnings"]:
        for w in validation_results["warnings"]:
            f.write(f"  - {w}\n")
    else:
        f.write("  None\n")

    f.write("\nKEY STATISTICS:\n")
    f.write(f"  Total aligned rows: {parquet_rows:,}\n")
    f.write(f"  Total TXT files:    {n_files:,}\n")
    f.write(f"  Total timestamps:   {total_ts:,}\n")
    f.write(f"  Mean confidence:    {conf_stats['mean']:.3f}\n")
    if "file_sizes" in validation_results["statistics"]:
        cov = validation_results["statistics"]["file_sizes"]["coverage_pct"]
        f.write(f"  Avg hourly coverage: {cov:.1f}% of full hour\n")
    f.write(f"  Chronological files: {quality_scores.get('chronological', 0.0):.1f}%\n")

print(f"Human-readable summary saved to: {summary_file}")

print("\n" + "=" * 100)
print("ULTIMATE VALIDATION COMPLETE.")
print("=" * 100)
