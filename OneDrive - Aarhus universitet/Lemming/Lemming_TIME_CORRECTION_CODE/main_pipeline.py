# =============================================================================
# Main RTC-TTN Time Alignment Pipeline
# =============================================================================
# Orchestrates the complete data processing workflow for correcting RTC clock
# drift using TTN (The Things Network) reference timestamps.
#
# Why this exists: Device RTC clocks drift over time. This pipeline aligns
# SD card measurements to accurate TTN timestamps, producing time-corrected
# output files.
#
# Works for any device (H1, C3, A1, B1, etc.) with Excel logging.
#
# Created by: Edevardt Johan Danielsen
# Last edited: 21/11/2025
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timezone
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_any_dtype

# Import our modules
import config
from data_loading import (
    SDCardDataLoader, TTNDataLoader, DataSegmenter, validate_loaded_data
)
from time_alignment import (
    TTNTimeMatcher, SegmentTimeAligner, compute_alignment_statistics
)
from excel_logger import ExcelLogger

# Global variable for TTN stats (used in Excel logging)
_ttn_stats_for_logging = {}


# -----------------------------------------------------------------------------
# setup_logging: Initialize logging system
# Creates log directory and configures both file and console output.
# Why: Enables tracking of pipeline progress and debugging issues.
# -----------------------------------------------------------------------------
def setup_logging():
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = config.LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.DEBUG if config.VERBOSE else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RTC-TTN Time Alignment Pipeline Starting")
    logger.info(f"Device: {config.DEVICE_ID}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    return logger


# -----------------------------------------------------------------------------
# load_data: PHASE 1 - Load all input data
# Loads SD card data and TTN reference data, validates temporal overlap.
# Why: Both data sources are needed - SD card has measurements, TTN has
# accurate timestamps to correct the drifting RTC clock.
# -----------------------------------------------------------------------------
def load_data(logger):
    global _ttn_stats_for_logging
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Loading Data")
    logger.info("=" * 80)
    
    # Load SD card data
    logger.info("\n--- Loading SD Card Data ---")
    logger.info(f"Using RTC timezone offset: UTC+{config.DEVICE_RTC_OFFSET // 60} ({config.DEVICE_RTC_OFFSET} minutes)")
    sd_loader = SDCardDataLoader(
        config.SD_CARD_DATA_DIR,
        config.RTC_FORMAT,
        rtc_offset_minutes=config.DEVICE_RTC_OFFSET
    )
    sd_df = sd_loader.load_all_txt_files()

    # Enforce processing window: drop SD data before TTN coverage (2025-03-25 UTC)
    cutoff_start = datetime(2025, 3, 25, tzinfo=timezone.utc)
    if len(sd_df) > 0:
        before_cutoff = sd_df[sd_df['rtc_time'] < cutoff_start]
        if len(before_cutoff) > 0:
            logger.info(
                f"Filtering out {len(before_cutoff):,} SD rows before cutoff {cutoff_start}."
            )
            sd_df = sd_df[sd_df['rtc_time'] >= cutoff_start].reset_index(drop=True)
        else:
            logger.info("No SD rows before cutoff date; no filtering applied.")
    
    if len(sd_df) == 0:
        raise ValueError("No SD card data could be loaded!")
    
    # Load TTN reference data
    logger.info("\n--- Loading TTN Reference Data ---")
    ttn_loader = TTNDataLoader(config.TTN_DATA_DIR)
    ttn_df = ttn_loader.load_ttn_reference()
    
    # Store TTN stats for Excel logging
    _ttn_stats_for_logging = {
        'total_points': len(ttn_df),
        'format_breakdown': ttn_df['format'].value_counts().to_dict() if 'format' in ttn_df.columns else {}
    }
    
    # Validate data
    logger.info("\n--- Validating Loaded Data ---")
    validation_results = validate_loaded_data(
        sd_df,
        ttn_df,
        (config.RTC_VALID_START, config.RTC_VALID_END)
    )
    
    # Log validation results
    logger.info("\nSD Card Data Quality:")
    for key, value in validation_results['sd_card'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nTTN Reference Quality:")
    for key, value in validation_results['ttn'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nData Overlap:")
    for key, value in validation_results['overlap'].items():
        logger.info(f"  {key}: {value}")
    
    if not validation_results['overlap']['has_overlap']:
        logger.error("ERROR: No temporal overlap between SD card and TTN data!")
        raise ValueError("SD card and TTN data do not overlap in time")
    
    # Save validation report
    if config.SAVE_INTERMEDIATE_FILES:
        validation_file = config.VALIDATION_DIR / "data_validation.json"
        config.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(validation_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            val_json = json.loads(
                json.dumps(validation_results, default=str)
            )
            json.dump(val_json, f, indent=2)
        
        logger.info(f"\nValidation report saved: {validation_file}")
    
    return sd_df, ttn_df


# -----------------------------------------------------------------------------
# segment_data: PHASE 2 - Segment data based on time gaps
# Splits continuous SD card data into segments separated by large gaps.
# Why: RTC drift correction works best within continuous segments. Large gaps
# (e.g., device off) require separate alignment anchors.
# -----------------------------------------------------------------------------
def segment_data(sd_df, logger):
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Segmenting Data")
    logger.info("=" * 80)
    
    segmenter = DataSegmenter(
        max_gap=config.MAX_INTRA_SEGMENT_GAP,
        min_segment_size=config.MIN_SEGMENT_SIZE
    )
    
    segments = segmenter.segment_data(sd_df)
    
    logger.info(f"\nCreated {len(segments)} segments")
    
    # Log segment details
    segment_summary = []
    for i, seg in enumerate(segments):
        summary = {
            'segment_id': i,
            'size': len(seg),
            'start_time': seg['rtc_time'].min(),
            'end_time': seg['rtc_time'].max(),
            'duration_minutes': (seg['rtc_time'].max() - seg['rtc_time'].min()).total_seconds() / 60,
            'num_files': seg['file_name'].nunique()
        }
        segment_summary.append(summary)
        logger.info(
            f"  Segment {i}: {len(seg)} lines, "
            f"{summary['duration_minutes']:.1f} minutes, "
            f"{summary['num_files']} files"
        )
    
    # Save segment summary
    if config.SAVE_INTERMEDIATE_FILES:
        seg_summary_df = pd.DataFrame(segment_summary)
        seg_file = config.ALIGNED_DATA_DIR / "segment_summary.csv"
        config.ALIGNED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        seg_summary_df.to_csv(seg_file, index=False)
        logger.info(f"\nSegment summary saved: {seg_file}")
    
    return segments


# -----------------------------------------------------------------------------
# align_segments: PHASE 3 - Align each segment to TTN reference times
# Matches SD card RTC times to accurate TTN timestamps using anchor points.
# Why: TTN timestamps are the ground truth. By finding matching points between
# SD card and TTN data, we can calculate and correct RTC drift.
# -----------------------------------------------------------------------------
def align_segments(segments, ttn_df, logger):
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: Aligning Segments to TTN Time")
    logger.info("=" * 80)
    
    # Initialize time matcher
    ttn_matcher = TTNTimeMatcher(ttn_df)
    aligner = SegmentTimeAligner(ttn_matcher)
    
    aligned_segments = []
    alignment_stats = []
    
    # Position anchor diagnostics
    pos_anchor_counts = {}

    for i, segment in enumerate(tqdm(segments, desc="Aligning segments")):
        logger.info(f"\n--- Processing Segment {i} ---")
        logger.info(f"  Size: {len(segment)} lines")
        logger.info(f"  RTC range: {segment['rtc_time'].min()} to {segment['rtc_time'].max()}")
        
        # Align segment
        aligned = aligner.align_segment(
            segment['rtc_time'],
            strategy='adaptive',
            positions=segment['concat_position'] if 'concat_position' in segment.columns else None
        )
        
        # Add segment metadata
        aligned['segment_id'] = i
        for col in ['file_path', 'file_name', 'line_number', 'line_content', 'data_values', 'concat_position', 'concat_group_id']:
            if col in segment.columns:
                aligned[col] = segment[col].values
        
        # Add original RTC for drift calculation
        aligned['original_rtc'] = segment['rtc_time'].values
        
        # Compute alignment statistics
        stats = compute_alignment_statistics(aligned)
        
        # Ensure all aligned times have timezone for logging
        aligned['aligned_time'] = pd.to_datetime(aligned['aligned_time'], utc=True)
        
        stats['segment_id'] = i
        stats['segment_size'] = len(segment)
        alignment_stats.append(stats)

        # Track anchors by concat_position
        if 'concat_position' in aligned.columns and 'confidence' in aligned.columns:
            high_conf = aligned[aligned['confidence'] >= 0.7]
            counts = high_conf['concat_position'].value_counts()
            for pos, cnt in counts.items():
                pos_anchor_counts[pos] = pos_anchor_counts.get(pos, 0) + int(cnt)
        
        logger.info(f"  Aligned time range: {aligned['aligned_time'].min()} to {aligned['aligned_time'].max()}")
        logger.info(f"  Mean confidence: {stats['mean_confidence']:.2f}")
        logger.info(f"  Mean drift: {stats['mean_drift_seconds']:.1f} seconds")
        
        aligned_segments.append(aligned)
    
    # Combine all segments
    all_aligned = pd.concat(aligned_segments, ignore_index=True)
    
    # Calculate drift for Excel logging
    if 'aligned_time' in all_aligned.columns and 'original_rtc' in all_aligned.columns:
        all_aligned['drift_seconds'] = (
            pd.to_datetime(all_aligned['aligned_time'], utc=True) -
            pd.to_datetime(all_aligned['original_rtc'], utc=True)
        ).dt.total_seconds()
        
    # Save aligned data
    if config.SAVE_INTERMEDIATE_FILES:
        aligned_file = config.ALIGNED_DATA_DIR / "aligned_data.parquet"
        all_aligned.to_parquet(aligned_file, index=False)
        logger.info(f"\nAligned data saved: {aligned_file}")
        
        stats_file = config.ALIGNED_DATA_DIR / "alignment_statistics.csv"
        pd.DataFrame(alignment_stats).to_csv(stats_file, index=False)
        logger.info(f"Alignment statistics saved: {stats_file}")
    
    # Overall statistics
    logger.info("\n--- Overall Alignment Statistics ---")
    logger.info(f"Total aligned lines: {len(all_aligned)}")
    logger.info(f"Mean confidence: {all_aligned['confidence'].mean():.3f}")
    logger.info(
        "High confidence (>=0.9): "
        f"{(all_aligned['confidence'] >= 0.9).sum()} "
        f"({(all_aligned['confidence'] >= 0.9).mean() * 100:.1f}%)"
    )
    logger.info(
        "Low confidence (<0.6): "
        f"{(all_aligned['confidence'] < 0.6).sum()} "
        f"({(all_aligned['confidence'] < 0.6).mean() * 100:.1f}%)"
    )

    # Log per-position anchor coverage
    if pos_anchor_counts:
        logger.info("\nAnchor counts by concat_position (confidence >= 0.7):")
        for pos, cnt in sorted(pos_anchor_counts.items()):
            logger.info(f"  pos={pos}: {cnt}")
    
    return all_aligned


# -----------------------------------------------------------------------------
# apply_group_anchor_corrections: Propagate drift within concatenated groups
# Within each concat_group_id (original physical line), if any row has good
# alignment, propagate its drift to all siblings.
# Why: When multiple measurements are concatenated from the same physical line,
# they should all share the same time correction to stay synchronized.
# -----------------------------------------------------------------------------
def apply_group_anchor_corrections(aligned_df: pd.DataFrame, logger) -> pd.DataFrame:
    if 'concat_group_id' not in aligned_df.columns:
        return aligned_df

    aligned_df = aligned_df.copy()
    conf_col = 'confidence' if 'confidence' in aligned_df.columns else 'alignment_confidence'
    updated_groups = 0
    updated_rows = 0

    # Ensure datetime types
    aligned_df['aligned_time'] = pd.to_datetime(aligned_df['aligned_time'], utc=True)
    aligned_df['original_rtc'] = pd.to_datetime(aligned_df['original_rtc'], utc=True)

    for gid, grp in aligned_df.groupby('concat_group_id', sort=False):
        if len(grp) <= 1:
            continue

        valid = grp.dropna(subset=['aligned_time'])
        if len(valid) == 0:
            continue

        # Pick the best anchor in the group
        anchor_idx = valid[conf_col].idxmax() if conf_col in valid.columns else valid.index[0]
        anchor_row = aligned_df.loc[anchor_idx]

        base_delta = anchor_row['aligned_time'] - anchor_row['original_rtc']
        group_mask = aligned_df['concat_group_id'] == gid

        aligned_df.loc[group_mask, 'aligned_time'] = (
            aligned_df.loc[group_mask, 'original_rtc'] + base_delta
        )
        if 'drift_seconds' in aligned_df.columns:
            aligned_df.loc[group_mask, 'drift_seconds'] = base_delta.total_seconds()

        updated_groups += 1
        updated_rows += group_mask.sum()

    if updated_groups > 0:
        logger.info(
            "Applied group anchor corrections to %d groups (%d rows).",
            updated_groups, updated_rows
        )

    return aligned_df


# -----------------------------------------------------------------------------
# _detect_ttn_time_column: Auto-detect TTN timestamp column
# Tries common column names (server_time, time_utc, etc.), falls back to
# the first datetime-like column.
# Why: TTN exports use different column names. This ensures flexibility
# without requiring manual column specification.
# -----------------------------------------------------------------------------
def _detect_ttn_time_column(ttn_df: pd.DataFrame, logger) -> str:
    candidate_names = [
        "server_time",
        "time_utc",
        "received_at",
        "timestamp",
        "time",
        "ttn_time",
    ]

    # 1) Prefer known names if they exist and are datetime-like
    for name in candidate_names:
        if name in ttn_df.columns and is_datetime64_any_dtype(ttn_df[name]):
            logger.info(f"Using TTN time column: '{name}' (matched known name)")
            return name

    # 2) Otherwise, pick the first datetime-like column
    datetime_cols = [
        c for c in ttn_df.columns
        if is_datetime64_any_dtype(ttn_df[c])
    ]

    if datetime_cols:
        logger.info(
            "Using TTN time column: '%s' (first datetime-like column)",
            datetime_cols[0],
        )
        return datetime_cols[0]

    # 3) If still nothing, give up with a clear error
    logger.error(
        "Could not find a datetime-like column in TTN dataframe. "
        "Columns are: %s",
        list(ttn_df.columns),
    )
    raise ValueError("No datetime-like column found in TTN TTN dataframe.")


# -----------------------------------------------------------------------------
# validate_correction_accuracy: PHASE 3.5 - Validate correction accuracy
# Performs TWO types of validation against TTN ground truth:
#   1. Direct Validation: Compare corrected times to actual TTN (nearest match)
#   2. Cross-Validation: Hide 20% of TTN points, test alignment accuracy
# Why: Quantifies how accurate the time corrections are. Reports MAE, RMSE,
# and percentile errors so you know if the correction can be trusted.
# -----------------------------------------------------------------------------
def validate_correction_accuracy(aligned_df, ttn_df, logger):
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3.5: CORRECTION ACCURACY VALIDATION")
    logger.info("=" * 80)
    
    # -------------------------------------------------------------------------
    # Detect TTN time column robustly (instead of hard-coded 'server_time')
    # -------------------------------------------------------------------------
    ttn_time_col = _detect_ttn_time_column(ttn_df, logger)
    
    # Ensure both dataframes have timezone-aware timestamps
    aligned_df = aligned_df.copy()
    aligned_df['aligned_time'] = pd.to_datetime(aligned_df['aligned_time'], utc=True)
    aligned_df['original_rtc'] = pd.to_datetime(aligned_df['original_rtc'], utc=True)
    
    ttn_df = ttn_df.copy()
    ttn_df[ttn_time_col] = pd.to_datetime(ttn_df[ttn_time_col], utc=True)
    
    # Confidence ranges (used in both direct & cross-validation)
    confidence_ranges = [
        ('High (0.7-1.0)', 0.7, 1.0),
        ('Medium (0.5-0.7)', 0.5, 0.7),
        ('Low (0.0-0.5)', 0.0, 0.5)
    ]
    
    # =============================================================================
    # PART 1: DIRECT VALIDATION (Compare corrected times to actual TTN)
    # =============================================================================
    
    logger.info("\n--- DIRECT VALIDATION: Comparing to TTN Ground Truth ---")
    
    logger.info(f"Comparing {len(ttn_df)} TTN reference points to corrected data using nearest match...")
    per_pos_stats = {}
    
    # Sort data for efficient nearest-neighbour join
    join_cols = ['aligned_time', 'original_rtc', 'confidence']
    if 'concat_position' in aligned_df.columns:
        join_cols.append('concat_position')
    if 'concat_group_id' in aligned_df.columns:
        join_cols.append('concat_group_id')

    aligned_for_join = aligned_df[join_cols].sort_values('aligned_time')
    ttn_for_join = ttn_df.sort_values(ttn_time_col)
    
    # Nearest merge within 30 seconds tolerance
    merged = pd.merge_asof(
        ttn_for_join,
        aligned_for_join,
        left_on=ttn_time_col,
        right_on='aligned_time',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=30)
    )
    
    # Drop rows with no match
    direct_validation_df = merged.dropna(subset=['aligned_time']).copy()
    
    if len(direct_validation_df) == 0:
        logger.warning("[WARNING] No matching points found for direct validation!")
    else:
        # Compute errors
        ttn_times = direct_validation_df[ttn_time_col]
        corrected_times = direct_validation_df['aligned_time']
        original_rtc_times = direct_validation_df['original_rtc']
        
        direct_validation_df['error_seconds'] = (corrected_times - ttn_times).dt.total_seconds()
        direct_validation_df['abs_error_seconds'] = direct_validation_df['error_seconds'].abs()
        direct_validation_df['rtc_error_seconds'] = (original_rtc_times - ttn_times).dt.total_seconds()
        direct_validation_df['correction_improvement'] = (
            direct_validation_df['rtc_error_seconds'].abs() -
            direct_validation_df['abs_error_seconds'].abs()
        )
        
        # Report overall statistics
        logger.info(
            f"\n[STATS] Direct Validation Results ({len(direct_validation_df)} validation points):"
        )
        logger.info(
            f"  Mean absolute error: "
            f"{direct_validation_df['abs_error_seconds'].mean():.1f} seconds"
        )
        logger.info(
            f"  Median absolute error: "
            f"{direct_validation_df['abs_error_seconds'].median():.1f} seconds"
        )
        logger.info(
            f"  RMSE: "
            f"{np.sqrt((direct_validation_df['error_seconds']**2).mean()):.1f} seconds"
        )
        logger.info(
            f"  95th percentile: "
            f"±{direct_validation_df['abs_error_seconds'].quantile(0.95):.1f} seconds"
        )
        logger.info(
            f"  Max error: "
            f"{direct_validation_df['abs_error_seconds'].max():.1f} seconds"
        )
        logger.info(
            f"  Mean correction improvement: "
            f"{direct_validation_df['correction_improvement'].mean():.1f} seconds"
        )
        
        # Report by confidence level
        logger.info("\n--- Accuracy by Confidence Level ---")
        
        for level_name, low, high in confidence_ranges:
            subset = direct_validation_df[
                (direct_validation_df['confidence'] >= low) &
                (direct_validation_df['confidence'] < high)
            ]
            
            if len(subset) > 0:
                logger.info(f"\n{level_name}:")
                logger.info(f"  N = {len(subset)} points ({len(subset)/len(direct_validation_df)*100:.1f}%)")
                logger.info(f"  Mean error: {subset['error_seconds'].mean():.1f} seconds")
                logger.info(f"  Std dev: {subset['error_seconds'].std():.1f} seconds")
                logger.info(f"  MAE: {subset['abs_error_seconds'].mean():.1f} seconds")
                logger.info(
                    "  95th percentile: "
                    f"±{subset['abs_error_seconds'].quantile(0.95):.1f} seconds"
                )
                logger.info(
                    f"  Max error: {subset['abs_error_seconds'].max():.1f} seconds"
                )

        # Per-position breakdown (concat_position)
        per_pos_stats = {}
        if 'concat_position' in direct_validation_df.columns:
            logger.info("\n--- Accuracy by concat_position ---")
            for pos, grp in direct_validation_df.groupby('concat_position'):
                per_pos_stats[int(pos)] = {
                    'n_points': int(len(grp)),
                    'mae_seconds': float(grp['abs_error_seconds'].mean()),
                    'rmse_seconds': float(np.sqrt((grp['error_seconds']**2).mean())) if len(grp) > 0 else None,
                    'median_seconds': float(grp['abs_error_seconds'].median())
                }
                logger.info(
                    f"  pos={pos}: N={len(grp)}, MAE={per_pos_stats[int(pos)]['mae_seconds']:.1f}s, "
                    f"median={per_pos_stats[int(pos)]['median_seconds']:.1f}s"
                )
        
        # Save direct validation results
        if config.SAVE_INTERMEDIATE_FILES:
            direct_val_file = config.VALIDATION_DIR / "direct_validation_results.csv"
            direct_validation_df.to_csv(direct_val_file, index=False)
            logger.info(f"\n[OK] Direct validation results saved: {direct_val_file}")
    
    # =============================================================================
    # PART 2: CROSS-VALIDATION (Test aligned times against held-out TTN points)
    # =============================================================================

    logger.info("\n--- CROSS-VALIDATION: Testing Alignment Accuracy ---")
    logger.info("This tests how well aligned_time matches held-out TTN ground truth")
    logger.info("(Validates interpolation accuracy between anchor points)\n")

    # Only do cross-validation if we have enough TTN points
    if len(ttn_df) < 50:
        logger.warning("[WARNING] Insufficient TTN points for cross-validation (need at least 50)")
        cross_validation_df = pd.DataFrame()
    else:
        # Split TTN into train (80%) and test (20%)
        # The test set represents TTN points we "pretend" weren't used for alignment
        ttn_train, ttn_test = train_test_split(
            ttn_df,
            test_size=0.2,
            random_state=42  # For reproducibility
        )

        logger.info(f"Split TTN data: {len(ttn_train)} anchors used, {len(ttn_test)} held out for testing")
        logger.info("Testing if aligned_time matches held-out TTN ground truth...")

        # FIXED APPROACH: Test the ALREADY ALIGNED times against held-out TTN
        # This avoids the reference frame mismatch bug where re-alignment with
        # reduced TTN caused systematic 1-hour errors due to timezone handling.
        #
        # The test answers: "Does our pipeline's aligned_time match TTN ground truth
        # at points that represent interpolated regions between anchors?"

        cv_errors = []

        # Sort aligned_df by aligned_time for efficient lookup
        aligned_for_cv = aligned_df[['aligned_time', 'original_rtc', 'confidence']].copy()
        aligned_for_cv['aligned_time'] = pd.to_datetime(aligned_for_cv['aligned_time'], utc=True)
        aligned_for_cv = aligned_for_cv.sort_values('aligned_time').reset_index(drop=True)

        for _, ttn_test_row in tqdm(
            ttn_test.iterrows(),
            total=len(ttn_test),
            desc="Cross-validation"
        ):
            ttn_test_time = pd.to_datetime(ttn_test_row[ttn_time_col], utc=True)

            # Find the aligned measurement closest to this TTN test time
            # This tests: "At this TTN timestamp, does our aligned_time match?"
            time_diffs = (aligned_for_cv['aligned_time'] - ttn_test_time).abs()
            closest_idx = time_diffs.idxmin()
            closest_diff = time_diffs[closest_idx]

            # Only count if we have a measurement within 30 seconds of the TTN time
            if closest_diff < pd.Timedelta(seconds=30):
                aligned_time = aligned_for_cv.loc[closest_idx, 'aligned_time']
                original_rtc = aligned_for_cv.loc[closest_idx, 'original_rtc']
                confidence = aligned_for_cv.loc[closest_idx, 'confidence']

                # Calculate alignment error: how far is aligned_time from TTN ground truth?
                error_seconds = (aligned_time - ttn_test_time).total_seconds()

                # Also calculate what the RTC error was before correction
                original_rtc_utc = pd.to_datetime(original_rtc, utc=True)
                rtc_error_seconds = (original_rtc_utc - ttn_test_time).total_seconds()

                cv_errors.append({
                    'ttn_ground_truth': ttn_test_time,
                    'aligned_time': aligned_time,
                    'original_rtc': original_rtc,
                    'error_seconds': error_seconds,
                    'abs_error_seconds': abs(error_seconds),
                    'rtc_error_before_correction': rtc_error_seconds,
                    'correction_improvement': abs(rtc_error_seconds) - abs(error_seconds),
                    'alignment_confidence': confidence
                })

        if not cv_errors:
            logger.warning("[WARNING] No matching points found for cross-validation!")
            cross_validation_df = pd.DataFrame()
        else:
            cross_validation_df = pd.DataFrame(cv_errors)

            # Report cross-validation results
            logger.info(
                f"\n[STATS] Cross-Validation Results ({len(cross_validation_df)} test points):"
            )
            logger.info(
                f"  Mean absolute error: "
                f"{cross_validation_df['abs_error_seconds'].mean():.1f} seconds"
            )
            logger.info(
                f"  Median absolute error: "
                f"{cross_validation_df['abs_error_seconds'].median():.1f} seconds"
            )
            logger.info(
                f"  RMSE: "
                f"{np.sqrt((cross_validation_df['error_seconds']**2).mean()):.1f} seconds"
            )
            logger.info(
                f"  95th percentile: "
                f"±{cross_validation_df['abs_error_seconds'].quantile(0.95):.1f} seconds"
            )
            logger.info(
                f"  Max error: "
                f"{cross_validation_df['abs_error_seconds'].max():.1f} seconds"
            )

            # Show improvement from correction
            mean_improvement = cross_validation_df['correction_improvement'].mean()
            logger.info(f"\n  Mean correction improvement: {mean_improvement:.1f} seconds")
            logger.info(
                f"  (Positive = alignment is better than raw RTC)"
            )

            # Report by alignment confidence
            logger.info("\n--- Cross-Validation Accuracy by Alignment Confidence ---")

            for level_name, low, high in confidence_ranges:
                subset = cross_validation_df[
                    (cross_validation_df['alignment_confidence'] >= low) &
                    (cross_validation_df['alignment_confidence'] < high)
                ]

                if len(subset) > 0:
                    logger.info(f"\n{level_name}:")
                    logger.info(f"  N = {len(subset)} points ({len(subset)/len(cross_validation_df)*100:.1f}%)")
                    logger.info(
                        f"  MAE: {subset['abs_error_seconds'].mean():.1f} seconds"
                    )
                    logger.info(
                        "  95th percentile: "
                        f"±{subset['abs_error_seconds'].quantile(0.95):.1f} seconds"
                    )

            # Save cross-validation results
            if config.SAVE_INTERMEDIATE_FILES:
                cv_file = config.VALIDATION_DIR / "cross_validation_results.csv"
                cross_validation_df.to_csv(cv_file, index=False)
                logger.info(f"\n[OK] Cross-validation results saved: {cv_file}")
    
    # =============================================================================
    # SUMMARY COMPARISON
    # =============================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if len(direct_validation_df) > 0:
        logger.info("\nDirect Validation (actual correction accuracy):")
        logger.info(f"  Validation points: {len(direct_validation_df)}")
        logger.info(
            "  Mean absolute error: "
            f"±{direct_validation_df['abs_error_seconds'].mean():.1f} seconds"
        )
        logger.info(
            "  95% within: "
            f"±{direct_validation_df['abs_error_seconds'].quantile(0.95):.1f} seconds"
        )
    
    if 'cross_validation_df' in locals() and len(cross_validation_df) > 0:
        logger.info("\nCross-Validation (alignment accuracy vs held-out TTN):")
        logger.info(f"  Test points: {len(cross_validation_df)}")
        logger.info(
            "  Mean absolute error: "
            f"±{cross_validation_df['abs_error_seconds'].mean():.1f} seconds"
        )
        logger.info(
            "  95% within: "
            f"±{cross_validation_df['abs_error_seconds'].quantile(0.95):.1f} seconds"
        )
        if 'correction_improvement' in cross_validation_df.columns:
            logger.info(
                "  Mean correction improvement: "
                f"{cross_validation_df['correction_improvement'].mean():.1f} seconds"
            )
    else:
        cross_validation_df = pd.DataFrame()

    # Create summary for reporting
    validation_summary = {
        'direct_validation': {
            'n_points': int(len(direct_validation_df)) if len(direct_validation_df) > 0 else 0,
            'mae_seconds': float(direct_validation_df['abs_error_seconds'].mean())
            if len(direct_validation_df) > 0 else None,
            'rmse_seconds': float(
                np.sqrt((direct_validation_df['error_seconds']**2).mean())
            ) if len(direct_validation_df) > 0 else None,
            'percentile_95': float(
                direct_validation_df['abs_error_seconds'].quantile(0.95)
            ) if len(direct_validation_df) > 0 else None
        },
        'by_concat_position': per_pos_stats if per_pos_stats else {},
        'cross_validation': {
            'n_points': int(len(cross_validation_df)) if len(cross_validation_df) > 0 else 0,
            'mae_seconds': float(cross_validation_df['abs_error_seconds'].mean())
            if len(cross_validation_df) > 0 else None,
            'rmse_seconds': float(
                np.sqrt((cross_validation_df['error_seconds']**2).mean())
            ) if len(cross_validation_df) > 0 else None,
            'percentile_95': float(
                cross_validation_df['abs_error_seconds'].quantile(0.95)
            ) if len(cross_validation_df) > 0 else None,
            'mean_correction_improvement': float(
                cross_validation_df['correction_improvement'].mean()
            ) if len(cross_validation_df) > 0 and 'correction_improvement' in cross_validation_df.columns else None
        }
    }
    
    # Save summary
    if config.SAVE_INTERMEDIATE_FILES:
        summary_file = config.VALIDATION_DIR / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        logger.info(f"\n[OK] Validation summary saved: {summary_file}")
    
    return direct_validation_df, cross_validation_df, validation_summary


# -----------------------------------------------------------------------------
# propagate_drift_backwards: Apply drift correction to early RTC-only periods
# After segment alignment, propagate the first well-anchored drift backwards
# into earlier periods that had no TTN overlap.
# How it works:
#   - Find earliest row with high confidence (>= 0.7)
#   - Use its drift (aligned_time - original_rtc) as reference
#   - Apply that drift to all earlier rows with low confidence
# Why: Early data before TTN coverage would otherwise remain uncorrected.
# This extrapolates the known drift backwards for better coverage.
# -----------------------------------------------------------------------------
def propagate_drift_backwards(aligned_df: pd.DataFrame, logger) -> pd.DataFrame:
    aligned_df = aligned_df.copy()

    # Basic sanity checks
    if 'aligned_time' not in aligned_df.columns or 'original_rtc' not in aligned_df.columns:
        logger.warning("Cannot propagate drift backwards: required columns missing.")
        return aligned_df

    if 'confidence' not in aligned_df.columns and 'alignment_confidence' not in aligned_df.columns:
        logger.warning("Cannot propagate drift backwards: no confidence column found.")
        return aligned_df

    # Handle either name: 'confidence' (current pipeline) or 'alignment_confidence'
    conf_col = 'confidence' if 'confidence' in aligned_df.columns else 'alignment_confidence'

    # Ensure datetime types with UTC
    aligned_df['aligned_time'] = pd.to_datetime(aligned_df['aligned_time'], utc=True)
    aligned_df['original_rtc'] = pd.to_datetime(aligned_df['original_rtc'], utc=True)

    def _propagate_for_subset(df_subset: pd.DataFrame) -> pd.DataFrame:
        # 1) Find earliest high-confidence point (>= 0.7)
        high_conf_mask = df_subset[conf_col] >= 0.7
        if not high_conf_mask.any():
            return df_subset

        earliest_idx = df_subset.loc[high_conf_mask, 'original_rtc'].idxmin()
        ref_row = df_subset.loc[earliest_idx]

        ref_delta = ref_row['aligned_time'] - ref_row['original_rtc']
        ref_delta_sec = ref_delta.total_seconds()

        logger.info(
            "Backward drift propagation (%s): using reference point at %s with drift %.1f seconds.",
            f"pos={ref_row.get('concat_position','NA')}", ref_row['original_rtc'], ref_delta_sec
        )

        # 2) Identify earlier rows that likely have no proper correction yet
        earlier_mask = df_subset['original_rtc'] < ref_row['original_rtc']
        low_conf_mask = df_subset[conf_col] < 0.6
        target_mask = earlier_mask & low_conf_mask

        if not target_mask.any():
            return df_subset

        n_rows = int(target_mask.sum())
        logger.info(
            "Applying backward drift of %.1f seconds to %d early rows (pos=%s).",
            ref_delta_sec, n_rows, ref_row.get('concat_position', 'NA')
        )

        df_subset.loc[target_mask, 'aligned_time'] = (
            df_subset.loc[target_mask, 'original_rtc'] + ref_delta
        )
        if 'drift_seconds' in df_subset.columns:
            df_subset.loc[target_mask, 'drift_seconds'] = ref_delta_sec
        return df_subset

    if 'concat_position' in aligned_df.columns:
        # Apply per position
        updated_parts = []
        for pos, subset in aligned_df.groupby('concat_position', sort=False):
            updated_parts.append(_propagate_for_subset(subset.copy()))
        aligned_df = pd.concat(updated_parts, ignore_index=True).sort_values('original_rtc')
    else:
        aligned_df = _propagate_for_subset(aligned_df)

    return aligned_df


# -----------------------------------------------------------------------------
# reconstruct_hourly_files: PHASE 4 - Reconstruct hourly .TXT files
# Creates output files with BOTH original and corrected timestamps.
# Output format:
#   - Header line with column names
#   - Original_RTC column (device time before correction)
#   - All data columns (sensor values)
#   - Corrected_Time column (aligned to TTN, spread for uniqueness)
#   - Corrected_Local_Time column (Copenhagen timezone)
# Why: Preserves original RTC for reference while providing corrected times.
# Files are organized by hour in month folders, matching original structure.
# -----------------------------------------------------------------------------
def reconstruct_hourly_files(aligned_df, logger):
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: Reconstructing Hourly Files")
    logger.info("=" * 80)
    
    # Create output directory structure
    config.RECONSTRUCTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort by aligned/spread time FIRST
    logger.info("Sorting data chronologically...")
    aligned_df = aligned_df.sort_values('aligned_time').reset_index(drop=True)

    # Drop exact repeats within the same concat_group_id to avoid ballooning coverage
    if 'concat_group_id' in aligned_df.columns:
        before = len(aligned_df)
        try:
            aligned_df['_vals_tuple'] = aligned_df['data_values'].apply(tuple)
            dedup_subset = ['concat_group_id', 'original_rtc', '_vals_tuple']
            aligned_df = aligned_df[~aligned_df.duplicated(subset=dedup_subset, keep='first')].copy()
            aligned_df = aligned_df.drop(columns=['_vals_tuple'])
            removed = before - len(aligned_df)
            if removed > 0:
                logger.info(f"Removed {removed:,} exact repeats within concat_group_id before spreading.")
        except Exception as e:
            logger.warning(f"Could not drop in-group exact repeats: {e}")
    
    # Check for reversals after sorting
    time_diffs = aligned_df['aligned_time'].diff().dt.total_seconds()
    reversals = (time_diffs < 0).sum()
    logger.info(f"Time reversals after sorting: {reversals}")
    if reversals > 0:
        logger.warning("[WARNING] Data still has reversals after sorting - check segment overlaps!")
    
    # Spread duplicate timestamps BEFORE grouping
    logger.info("\nDetecting and spreading duplicate timestamps...")
    
    # Find all duplicate timestamps
    duplicate_mask = aligned_df.duplicated('aligned_time', keep=False)
    num_duplicates = duplicate_mask.sum()
    logger.info(
        f"Found {num_duplicates:,} rows with duplicate timestamps "
        f"({num_duplicates/len(aligned_df)*100:.1f}%)"
    )
    
    if num_duplicates > 0:
# Get device-specific sampling interval
        device_interval = config.DEVICE_SAMPLE_INTERVAL_SECONDS
        logger.info(f"Spreading duplicates using original RTC intervals with {device_interval}s fallback (device {config.DEVICE_ID})...")

        spread_times = []
        prev_aligned_time = None
        prev_original_rtc = None

        for idx, row in aligned_df.iterrows():
            aligned_time = row['aligned_time']
            original_rtc = row['original_rtc']
            
            if prev_aligned_time is None:
                # First row
                spread_times.append(aligned_time)
            elif aligned_time == prev_aligned_time:
                # Duplicate - try to use original RTC interval
                rtc_interval = (original_rtc - prev_original_rtc).total_seconds()
                
                # Sanity check: if interval is unreasonable, use device median
                if 2 <= rtc_interval <= 20:  # Reasonable range
                    new_time = spread_times[-1] + pd.Timedelta(seconds=rtc_interval)
                else:
                    # Fallback to median interval
                    new_time = spread_times[-1] + pd.Timedelta(seconds=device_interval)
                
                spread_times.append(new_time)
            else:
                # New unique timestamp
                spread_times.append(aligned_time)
            
            prev_aligned_time = aligned_time
            prev_original_rtc = original_rtc
                
        # Replace aligned_time with spread times
        aligned_df['spread_time'] = spread_times
        aligned_df['spread_time'] = pd.to_datetime(aligned_df['spread_time'], utc=True)
        
        # Verify spreading worked
        final_duplicates = aligned_df.duplicated('spread_time', keep=False).sum()
        logger.info(
            f"After spreading: {final_duplicates:,} duplicates remaining "
            f"({final_duplicates/len(aligned_df)*100:.1f}%)"
        )
        
        # Use spread_time for file organization
        time_col = 'spread_time'
    else:
        logger.info("No duplicates found - using aligned_time as-is")
        aligned_df['spread_time'] = aligned_df['aligned_time']
        time_col = 'spread_time'
    
    # Group by hour AFTER spreading
    logger.info("\nGrouping by hour...")
    # IMPORTANT: Generate filenames from LOCAL time, not UTC!
    # Convert spread_time (UTC) to Copenhagen local for filename generation
    local_time_for_filenames = aligned_df[time_col].dt.tz_convert('Europe/Copenhagen')
    aligned_df['hour_key'] = local_time_for_filenames.dt.strftime('%y%m%d%H')
    aligned_df['month_folder'] = local_time_for_filenames.dt.strftime('%m')
    
    files_written = {}
    
    for (month, hour_key), group in tqdm(
        aligned_df.groupby(['month_folder', 'hour_key']),
        desc="Writing hourly files"
    ):
        # Create month folder
        month_dir = config.RECONSTRUCTED_DATA_DIR / month
        month_dir.mkdir(exist_ok=True)
        
        # Filename
        filename = f"{hour_key}.TXT"
        filepath = month_dir / filename
        
        # Sort by spread time within hour
        group = group.sort_values(time_col).reset_index(drop=True)
        
# DYNAMIC COLUMN DETECTION: Find max columns in this hour
        max_columns = group['data_values'].apply(len).max()
        
        # Generate appropriate header based on column count
        if max_columns <= 5:
            # OLD format: 5 data columns
            data_headers = ["Sensor_Volt", "CH4_Raw", "CO2_ppm", "Temp_C", "RH_pct"]
        elif max_columns <= 6:
            # OLD format with 6 columns (rare, some devices)
            data_headers = ["Sensor_Volt", "CH4_Raw", "CO2_ppm", "Temp_C", "RH_pct", "Extra1"]
        else:
            # NEW format with OPTOD: 10 data columns
            data_headers = ["Sensor_Volt", "CH4_Raw", "CO2_ppm", "Temp_C", "RH_pct", 
                          "DO_mgL", "DO_sat_pct", "Topt_C", "Extra4", "Sal"]

        # Ensure we have enough headers (in case there are even more columns)
        while len(data_headers) < max_columns:
            data_headers.append(f"Extra{len(data_headers)-5}")

        # Write file with DYNAMIC headers and padding
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header line with proper column names
            header = (
                "Original_RTC," +
                ",".join(data_headers[:max_columns]) +
                ",Corrected_Time,Corrected_Local_Time\n"
            )
            f.write(header)

            # Write data lines
            for _, row in group.iterrows():
                # Original RTC timestamp
                original_rtc_str = pd.to_datetime(row['original_rtc'], utc=True).strftime(config.OUTPUT_FORMAT)

                # Data values (pad to max_columns if needed)
                data_values = list(row['data_values'])
                if len(data_values) < max_columns:
                    data_values = data_values + [''] * (max_columns - len(data_values))

                # Corrected timestamps (UTC and local Copenhagen time)
                corrected_ts = row[time_col]
                if corrected_ts.tzinfo is None:
                    corrected_ts = corrected_ts.tz_localize('UTC')
                corrected_time_str = corrected_ts.strftime(config.OUTPUT_FORMAT)
                corrected_local_str = corrected_ts.tz_convert('Europe/Copenhagen').strftime(config.OUTPUT_FORMAT)

                new_line = (
                    f"{original_rtc_str}," +
                    ",".join(map(str, data_values[:max_columns])) +
                    f",{corrected_time_str},{corrected_local_str}\n"
                )
                f.write(new_line)
                       
        files_written[filename] = {
            'path': str(filepath),
            'lines': len(group),  # data lines only (headers excluded)
            'time_range': (group[time_col].min(), group[time_col].max()),
            'mean_confidence': group['confidence'].mean()
        }
    
    logger.info(f"\n[OK] Wrote {len(files_written)} hourly files with dual timestamps")
    
    # Statistics
    files_df = pd.DataFrame([
        {'filename': k, **v} for k, v in files_written.items()
    ])
    expected_samples = 3600 // config.DEVICE_SAMPLE_INTERVAL_SECONDS
    logger.info(f"Mean lines per file: {files_df['lines'].mean():.0f} (excluding header)")    
    logger.info(f"Expected (full hour, {config.DEVICE_SAMPLE_INTERVAL_SECONDS}s interval): {expected_samples}")
    logger.info(f"Coverage: {files_df['lines'].mean()/expected_samples*100:.1f}%")
 
    # Save file manifest
    manifest_file = config.RECONSTRUCTED_DATA_DIR / "file_manifest.csv"
    manifest_data = []
    for filename, info in files_written.items():
        manifest_data.append({
            'filename': filename,
            'lines': info['lines'],
            'start_time': info['time_range'][0],
            'end_time': info['time_range'][1],
            'mean_confidence': info['mean_confidence']
        })
    
    pd.DataFrame(manifest_data).to_csv(manifest_file, index=False)
    logger.info(f"File manifest saved: {manifest_file}")
    
    return files_written


# -----------------------------------------------------------------------------
# validate_output: PHASE 5 - Validate reconstructed files
# Reads back written files and checks for:
#   - Chronological timestamp order
#   - Time reversals (should be none)
#   - Large gaps (> 5 minutes)
#   - Correct hour assignment
#   - Mean drift statistics
# Why: Ensures output quality and catches any issues in the reconstruction.
# Flags problematic files for manual review if needed.
# -----------------------------------------------------------------------------
def validate_output(files_written, aligned_df, logger):
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: Validating Output")
    logger.info("=" * 80)
    
    validation_results = []
    
    for filename, info in tqdm(files_written.items(), desc="Validating files"):
        filepath = Path(info['path'])
        
        # Read file back
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header line
            data_lines = lines[1:] if len(lines) > 1 else []
            
            if len(data_lines) == 0:
                logger.warning(f"File {filename} has no data lines")
                continue
            
            timestamps = []
            original_timestamps = []
            
            for line in data_lines:
                try:
                    parts = line.strip().split(',')
                    # Expected: Original_RTC, Data1..N, Corrected_Time
                    if len(parts) >= 3:
                        original_ts_str = parts[0]
                        corrected_ts_str = parts[-1]
                        
                        original_ts = datetime.strptime(original_ts_str, config.OUTPUT_FORMAT)
                        corrected_ts = datetime.strptime(corrected_ts_str, config.OUTPUT_FORMAT)
                        
                        original_timestamps.append(original_ts)
                        timestamps.append(corrected_ts)
                except Exception:
                    # Skip malformed lines
                    pass
            
            if len(timestamps) < 2:
                logger.warning(f"File {filename} has insufficient valid timestamps")
                continue
            
            ts_series = pd.Series(timestamps)
            
            # Check chronological order (corrected timestamps)
            timestamps_sorted = sorted(timestamps)
            is_chronological = timestamps == timestamps_sorted
            
            # Check for time reversals
            time_diffs = ts_series.diff().dt.total_seconds()
            num_reversals = (time_diffs < 0).sum()
            
            # Check time gaps
            max_gap_seconds = time_diffs.max()
            gaps_over_threshold = (time_diffs > config.MAX_ALLOWED_TIME_GAP_MINUTES * 60).sum()
            
            # Expected hour from filename
            expected_hour = int(filename.replace('.TXT', '')[-2:])
            actual_hours = ts_series.dt.hour
            correct_hour_pct = (actual_hours == expected_hour).mean() * 100
            
            # Mean drift (difference between original and corrected)
            if len(original_timestamps) == len(timestamps):
                drifts = [
                    (corrected - original).total_seconds()
                    for original, corrected in zip(original_timestamps, timestamps)
                ]
                mean_drift_seconds = float(np.mean(drifts))
            else:
                mean_drift_seconds = None
            
            validation_results.append({
                'filename': filename,
                'num_lines': len(data_lines),
                'num_valid_timestamps': len(timestamps),
                'is_chronological': is_chronological,
                'num_time_reversals': int(num_reversals),
                'max_gap_seconds': float(max_gap_seconds),
                'gaps_over_5min': int(gaps_over_threshold),
                'correct_hour_pct': float(correct_hour_pct),
                'expected_hour': expected_hour,
                'mean_drift_seconds': mean_drift_seconds
            })
        
        except Exception as e:
            logger.error(f"Error validating {filename}: {e}")
            validation_results.append({
                'filename': filename,
                'error': str(e)
            })
    
    # Save validation results
    val_df = pd.DataFrame(validation_results)
    val_file = config.VALIDATION_DIR / "output_validation.csv"
    val_df.to_csv(val_file, index=False)
    logger.info(f"\nValidation results saved: {val_file}")
    
    # Summary statistics
    logger.info("\n--- Validation Summary ---")
    if 'is_chronological' in val_df.columns:
        logger.info(
            "Files with chronological timestamps: "
            f"{val_df['is_chronological'].sum()} / {len(val_df)}"
        )
        logger.info(
            "Files with time reversals: "
            f"{(val_df['num_time_reversals'] > 0).sum()}"
        )
        logger.info(
            "Files with gaps > 5 min: "
            f"{(val_df['gaps_over_5min'] > 0).sum()}"
        )
        logger.info(
            "Mean correct hour percentage: "
            f"{val_df['correct_hour_pct'].mean():.1f}%"
        )
        
        if 'mean_drift_seconds' in val_df.columns and val_df['mean_drift_seconds'].notna().any():
            mean_drift = val_df['mean_drift_seconds'].dropna().mean()
            logger.info(
                "Mean RTC drift across all files: "
                f"{mean_drift:.1f} seconds ({mean_drift/3600:.2f} hours)"
            )
    
    # Identify problematic files
    if 'is_chronological' in val_df.columns:
        problematic = val_df[
            (~val_df['is_chronological']) |
            (val_df['num_time_reversals'] > 0) |
            (val_df['correct_hour_pct'] < 80)
        ]
        
        if len(problematic) > 0:
            logger.warning(f"\n{len(problematic)} files have potential issues:")
            for _, row in problematic.head(10).iterrows():
                logger.warning(
                    f"  {row['filename']}: "
                    f"reversals={row.get('num_time_reversals', 'N/A')}, "
                    f"correct_hour={row.get('correct_hour_pct', 'N/A'):.1f}%"
                )
    
    return val_df


# -----------------------------------------------------------------------------
# generate_summary_report: Generate final summary and log to Excel
# Creates JSON summary report with:
#   - Processing timestamp and device ID
#   - Input statistics (total lines, date range)
#   - Alignment quality metrics (confidence, drift)
#   - Validation accuracy (MAE, RMSE from cross-validation)
#   - Output statistics (files created, lines written)
# Also logs to Excel logbook if ENABLE_EXCEL_LOGGING is True in config.
# Why: Provides complete record of each processing run for documentation.
# -----------------------------------------------------------------------------
def generate_summary_report(aligned_df, val_df, validation_summary, logger):
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY REPORT")
    logger.info("=" * 80)
    
    processing_timestamp = datetime.now()
    
    # SD card stats
    sd_card_stats = {
        'total_lines': len(aligned_df),
        'date_range': (
            aligned_df['original_rtc'].min() if 'original_rtc' in aligned_df.columns else None,
            aligned_df['original_rtc'].max() if 'original_rtc' in aligned_df.columns else None
        ),
        'unique_files': aligned_df['file_name'].nunique() if 'file_name' in aligned_df.columns else 0
    }
    
    # TTN stats (from global variable)
    ttn_stats = _ttn_stats_for_logging
    
    # Alignment stats
    mean_drift_seconds = aligned_df['drift_seconds'].mean() if 'drift_seconds' in aligned_df.columns else 0
    max_drift_seconds = aligned_df['drift_seconds'].max() if 'drift_seconds' in aligned_df.columns else 0
    
    alignment_stats = {
        'mean_confidence': float(aligned_df['confidence'].mean()),
        'mean_drift_days': mean_drift_seconds / 86400 if mean_drift_seconds is not None else None,
        'max_drift_days': max_drift_seconds / 86400 if max_drift_seconds is not None else None,
        'mean_drift_seconds': mean_drift_seconds,
        'max_drift_seconds': max_drift_seconds,
        'high_confidence_pct': float((aligned_df['confidence'] >= 0.9).mean() * 100),
        'medium_confidence_pct': float(
            ((aligned_df['confidence'] >= 0.6) &
             (aligned_df['confidence'] < 0.9)).mean() * 100
        ),
        'low_confidence_pct': float((aligned_df['confidence'] < 0.6).mean() * 100)
    }
    
    # Output stats
    output_stats = {
        'files_created': len(val_df),
        'total_output_lines': len(aligned_df)
    }
    
    # Validation stats
    validation_stats = None
    if 'is_chronological' in val_df.columns:
        validation_stats = {
            'chronological_files': int(val_df['is_chronological'].sum()),
            'files_with_reversals': int((val_df['num_time_reversals'] > 0).sum()),
            'mean_correct_hour_pct': float(val_df['correct_hour_pct'].mean())
        }
    
    # Create JSON report
    report = {
        'processing_timestamp': processing_timestamp.isoformat(),
        'device_id': config.DEVICE_ID,
        'input_statistics': {
            'total_input_lines': sd_card_stats['total_lines'],
            'date_range': (
                str(sd_card_stats['date_range'][0]),
                str(sd_card_stats['date_range'][1])
            )
        },
        'alignment_quality': alignment_stats,
        'validation_accuracy': validation_summary,  # Include validation results
        'output_statistics': output_stats
    }
    
    if validation_stats:
        report['output_quality'] = validation_stats
    
    # Save JSON report
    report_file = config.OUTPUT_DIR / "summary_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("\nProcessing complete!")
    logger.info(f"Summary report saved: {report_file}")
    logger.info(f"\nOutput directory: {config.RECONSTRUCTED_DATA_DIR}")
    
    # Print validation summary
    if (
        validation_summary.get('direct_validation') and
        validation_summary['direct_validation'].get('n_points', 0) > 0
    ):
        logger.info("\n[STATS] Validation Accuracy Summary:")
        logger.info(
            "  Direct validation MAE: "
            f"±{validation_summary['direct_validation']['mae_seconds']:.1f} seconds"
        )
        if validation_summary['cross_validation'].get('n_points', 0) > 0:
            logger.info(
                "  Cross-validation MAE: "
                f"±{validation_summary['cross_validation']['mae_seconds']:.1f} seconds"
            )
    
    # Excel logging
    if config.ENABLE_EXCEL_LOGGING:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("LOGGING TO EXCEL")
            logger.info("=" * 80)
            
            excel_logger = ExcelLogger(config.LOGBOOK_FILE)
            
            # Determine issues
            issues = "None"
            if validation_stats:
                if validation_stats['files_with_reversals'] > 0:
                    issues = f"{validation_stats['files_with_reversals']} files with time reversals"
                elif validation_stats['chronological_files'] < len(val_df):
                    issues = f"{len(val_df) - validation_stats['chronological_files']} non-chronological files"
            
            excel_logger.log_processing_run(
                device_id=config.DEVICE_ID,
                processing_timestamp=processing_timestamp,
                sd_card_stats=sd_card_stats,
                ttn_stats=ttn_stats,
                alignment_stats=alignment_stats,
                output_stats=output_stats,
                validation_stats=validation_stats,
                issues=issues
            )
            
            logger.info(f"[OK] Excel log updated: {config.LOGBOOK_FILE}")
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to log to Excel: {e}")
            logger.error("Pipeline completed successfully but Excel logging failed")
    
    return report


# -----------------------------------------------------------------------------
# main: Main pipeline execution
# Orchestrates all phases in sequence:
#   Phase 1: Load SD card and TTN data
#   Phase 2: Segment data by time gaps
#   Phase 3: Align segments to TTN reference times
#   Phase 3.5: Validate correction accuracy
#   Phase 4: Reconstruct hourly output files
#   Phase 5: Validate output files
#   Final: Generate summary report and Excel log
# Why: Single entry point that runs the complete time correction workflow.
# -----------------------------------------------------------------------------
def main():
    logger = setup_logging()
    try:
        # Phase 1: Load data
        sd_df, ttn_df = load_data(logger)
        
        # Phase 2: Segment data
        segments = segment_data(sd_df, logger)
        
        # Phase 3: Align segments
        aligned_df = align_segments(segments, ttn_df, logger)

        # Backward propagation for early RTC-only periods (no TTN overlap)
        logger.info(
            "\nApplying backward drift propagation to early RTC-only periods..."
        )
        aligned_df = propagate_drift_backwards(aligned_df, logger)

        # Apply group-level corrections so concatenated measurements move together
        logger.info("\nApplying group-level anchor corrections (concat_group_id)...")
        aligned_df = apply_group_anchor_corrections(aligned_df, logger)

        # Optional: recompute global alignment statistics after propagation
        try:
            updated_stats = compute_alignment_statistics(aligned_df)
            logger.info(
                "After backward propagation: mean drift = %.1f s, max drift = %.1f s",
                updated_stats.get('mean_drift_seconds', 0.0),
                updated_stats.get('max_drift_seconds', 0.0),
            )
        except Exception as e:
            logger.warning(f"Could not recompute alignment statistics after propagation: {e}")
        
        # NEW Phase 3.5: Validate correction accuracy
        direct_val_df, cross_val_df, validation_summary = validate_correction_accuracy(
            aligned_df, ttn_df, logger
        )
        
        # Phase 4: Reconstruct files (now with dual timestamps)
        files_written = reconstruct_hourly_files(aligned_df, logger)
        
        # Phase 5: Validate output
        val_df = validate_output(files_written, aligned_df, logger)
        
        # Generate final report (now includes validation accuracy)
        report = generate_summary_report(aligned_df, val_df, validation_summary, logger)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return report
    
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("PIPELINE FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'='*80}")
        raise


if __name__ == "__main__":
    main()


