# =============================================================================
# Post-Processing Validation and Analysis
# =============================================================================
# Comprehensive checks and visualizations for pipeline output.
#
# Why this exists: After the alignment pipeline runs, we need to validate
# the output files and analyze alignment quality. This module provides
# validation checks for reconstructed files and diagnostic analysis tools.
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta
import logging

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# OutputValidator: Validates reconstructed hourly files
# Checks chronological order, time gaps, correct hour, sample counts, etc.
# -----------------------------------------------------------------------------
class OutputValidator:

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = []
    
    # Perform comprehensive validation on all output files
    def validate_all_files(self):
        logger.info("Starting comprehensive validation...")
        
        txt_files = list(self.output_dir.rglob("*.TXT"))
        logger.info(f"Found {len(txt_files)} files to validate")
        
        for filepath in txt_files:
            result = self._validate_single_file(filepath)
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    # Validate a single file for timestamps, gaps, chronological order, etc.
    def _validate_single_file(self, filepath: Path) -> dict:
        result = {
            'filename': filepath.name,
            'path': str(filepath),
            'month': filepath.parent.name
        }
        
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            result['num_lines'] = len(lines)
            
            if len(lines) == 0:
                result['status'] = 'EMPTY'
                return result
            
            # Parse all timestamps and data
            timestamps = []
            data_fields = []
            
            for i, line in enumerate(lines):
                try:
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    
                    # Parse timestamp (last field)
                    ts_str = parts[-1].strip()
                    ts = pd.to_datetime(ts_str, format=config.OUTPUT_FORMAT)
                    timestamps.append(ts)
                    
                    # Count data fields
                    data_fields.append(len(parts) - 1)
                
                except Exception as e:
                    logger.debug(f"Error parsing line {i} in {filepath.name}: {e}")
            
            if len(timestamps) == 0:
                result['status'] = 'NO_VALID_TIMESTAMPS'
                return result
            
            result['num_valid_timestamps'] = len(timestamps)
            
            # Convert to series for analysis
            ts_series = pd.Series(timestamps)
            
            # Check 1: Chronological order
            is_sorted = ts_series.is_monotonic_increasing
            result['is_chronological'] = is_sorted
            
            # Check 2: Time reversals
            time_diffs = ts_series.diff().dt.total_seconds()
            reversals = time_diffs[time_diffs < 0]
            result['num_reversals'] = len(reversals)
            
            if len(reversals) > 0:
                result['max_reversal_seconds'] = abs(reversals.min())
            
            # Check 3: Time gaps
            result['max_gap_seconds'] = time_diffs.max()
            result['mean_gap_seconds'] = time_diffs.mean()
            result['median_gap_seconds'] = time_diffs.median()
            
            gaps_over_5min = time_diffs[time_diffs > 300].dropna()
            result['num_gaps_over_5min'] = len(gaps_over_5min)
            
            # Check 4: Expected hour
            expected_hour = int(filepath.stem[-2:])
            actual_hours = ts_series.dt.hour
            correct_hour_count = (actual_hours == expected_hour).sum()
            result['expected_hour'] = expected_hour
            result['correct_hour_count'] = correct_hour_count
            result['correct_hour_pct'] = (correct_hour_count / len(timestamps)) * 100
            
            # Check 5: Time range
            result['start_time'] = ts_series.min()
            result['end_time'] = ts_series.max()
            result['duration_minutes'] = (ts_series.max() - ts_series.min()).total_seconds() / 60
            
            # Check 6: Data field consistency
            if data_fields:
                result['min_fields'] = min(data_fields)
                result['max_fields'] = max(data_fields)
                result['inconsistent_fields'] = len(set(data_fields)) > 1
            
            # Check 7: Expected sample count
            # Device-specific interval
            expected_samples = 3600 // config.DEVICE_SAMPLE_INTERVAL_SECONDS
            result['expected_samples'] = expected_samples
            result['sample_count_deviation'] = len(timestamps) - expected_samples
            
            # Overall status
            issues = []
            if not is_sorted:
                issues.append('NOT_CHRONOLOGICAL')
            if len(reversals) > 0:
                issues.append(f'{len(reversals)}_REVERSALS')
            if len(gaps_over_5min) > 0:
                issues.append(f'{len(gaps_over_5min)}_LARGE_GAPS')
            if correct_hour_count / len(timestamps) < 0.8:
                issues.append('WRONG_HOUR')
            # Allow 20% deviation from expected count
            expected_samples = 3600 // config.DEVICE_SAMPLE_INTERVAL_SECONDS
            if abs(len(timestamps) - expected_samples) > expected_samples * 0.2:
                issues.append('UNEXPECTED_COUNT')
            
            result['status'] = 'OK' if len(issues) == 0 else '; '.join(issues)
            result['num_issues'] = len(issues)
        
        except Exception as e:
            result['status'] = f'ERROR: {str(e)}'
            logger.error(f"Error validating {filepath.name}: {e}")
        
        return result
    
    # Generate summary report from validation results
    def generate_report(self, results_df: pd.DataFrame) -> dict:
        report = {
            'total_files': len(results_df),
            'ok_files': (results_df['status'] == 'OK').sum(),
            'files_with_issues': (results_df['status'] != 'OK').sum()
        }
        
        if 'num_lines' in results_df.columns:
            report['total_lines'] = results_df['num_lines'].sum()
            report['mean_lines_per_file'] = results_df['num_lines'].mean()
        
        if 'is_chronological' in results_df.columns:
            report['chronological_files'] = results_df['is_chronological'].sum()
            report['files_with_reversals'] = (results_df['num_reversals'] > 0).sum()
            report['files_with_large_gaps'] = (results_df['num_gaps_over_5min'] > 0).sum()
        
        if 'correct_hour_pct' in results_df.columns:
            report['mean_correct_hour_pct'] = results_df['correct_hour_pct'].mean()
            report['files_mostly_correct_hour'] = (results_df['correct_hour_pct'] >= 80).sum()
        
        return report


# -----------------------------------------------------------------------------
# AlignmentAnalyzer: Analyzes alignment quality from intermediate data
# Computes drift patterns, confidence distributions, and creates diagnostic plots
# -----------------------------------------------------------------------------
class AlignmentAnalyzer:

    def __init__(self, aligned_data_path: Path):
        self.aligned_data_path = aligned_data_path
        self.df = None
    
    # Load aligned data from parquet file
    def load_data(self):
        logger.info(f"Loading aligned data from {self.aligned_data_path}")
        self.df = pd.read_parquet(self.aligned_data_path)
        logger.info(f"Loaded {len(self.df)} aligned records")
    
    # Analyze how RTC drift varies over time
    def analyze_drift_patterns(self):
        if self.df is None:
            self.load_data()
        
        self.df['drift_seconds'] = (
            self.df['aligned_time'] - self.df['original_rtc']
        ).dt.total_seconds()
        
        stats = {
            'mean_drift': self.df['drift_seconds'].mean(),
            'median_drift': self.df['drift_seconds'].median(),
            'std_drift': self.df['drift_seconds'].std(),
            'min_drift': self.df['drift_seconds'].min(),
            'max_drift': self.df['drift_seconds'].max(),
            'drift_range': self.df['drift_seconds'].max() - self.df['drift_seconds'].min()
        }
        
        return stats
    
    # Analyze confidence score distribution
    def analyze_confidence_distribution(self):
        if self.df is None:
            self.load_data()
        
        stats = {
            'mean_confidence': self.df['confidence'].mean(),
            'median_confidence': self.df['confidence'].median(),
            'high_confidence_pct': (self.df['confidence'] >= config.CONFIDENCE_HIGH).mean() * 100,
            'medium_confidence_pct': (
                (self.df['confidence'] >= config.CONFIDENCE_MEDIUM) &
                (self.df['confidence'] < config.CONFIDENCE_HIGH)
            ).mean() * 100,
            'low_confidence_pct': (self.df['confidence'] < config.CONFIDENCE_MEDIUM).mean() * 100
        }
        
        return stats
    
    # Create diagnostic visualizations (drift over time, confidence dist, etc.)
    def create_diagnostic_plots(self, output_dir: Path):
        if self.df is None:
            self.load_data()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Drift over time
        plt.figure(figsize=(14, 6))
        plt.plot(self.df['original_rtc'], self.df['drift_seconds'], 
                alpha=0.3, linewidth=0.5)
        plt.xlabel('RTC Time')
        plt.ylabel('Drift (seconds)')
        plt.title('RTC Drift Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'drift_over_time.png', dpi=300)
        plt.close()
        
        # Plot 2: Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['confidence'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Score Distribution')
        plt.axvline(config.CONFIDENCE_HIGH, color='green', 
                   linestyle='--', label='High confidence threshold')
        plt.axvline(config.CONFIDENCE_MEDIUM, color='orange', 
                   linestyle='--', label='Medium confidence threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300)
        plt.close()
        
        # Plot 3: RTC vs Aligned time scatter
        sample = self.df.sample(min(10000, len(self.df)))
        plt.figure(figsize=(10, 10))
        plt.scatter(sample['original_rtc'], sample['aligned_time'], 
                   alpha=0.1, s=1)
        
        # Add perfect alignment line
        min_time = min(sample['original_rtc'].min(), sample['aligned_time'].min())
        max_time = max(sample['original_rtc'].max(), sample['aligned_time'].max())
        plt.plot([min_time, max_time], [min_time, max_time], 
                'r--', label='Perfect alignment')
        
        plt.xlabel('Original RTC Time')
        plt.ylabel('Aligned TTN Time')
        plt.title('RTC vs TTN Time Alignment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'rtc_vs_ttn.png', dpi=300)
        plt.close()
        
        # Plot 4: Segment-wise confidence
        if 'segment_id' in self.df.columns:
            seg_conf = self.df.groupby('segment_id')['confidence'].mean()
            plt.figure(figsize=(12, 6))
            plt.bar(seg_conf.index, seg_conf.values)
            plt.xlabel('Segment ID')
            plt.ylabel('Mean Confidence')
            plt.title('Mean Confidence Score by Segment')
            plt.axhline(config.CONFIDENCE_HIGH, color='green', 
                       linestyle='--', alpha=0.5)
            plt.axhline(config.CONFIDENCE_MEDIUM, color='orange', 
                       linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'segment_confidence.png', dpi=300)
            plt.close()
        
        logger.info(f"Diagnostic plots saved to {output_dir}")


# Run comprehensive validation and analysis
def main():
    logger.info("="*80)
    logger.info("POST-PROCESSING VALIDATION AND ANALYSIS")
    logger.info("="*80)
    
    # Validate output files
    logger.info("\n--- Validating Output Files ---")
    validator = OutputValidator(config.RECONSTRUCTED_DATA_DIR)
    validation_df = validator.validate_all_files()
    
    # Save validation results
    val_file = config.VALIDATION_DIR / "comprehensive_validation.csv"
    validation_df.to_csv(val_file, index=False)
    logger.info(f"Validation results saved: {val_file}")
    
    # Generate report
    report = validator.generate_report(validation_df)
    logger.info("\n--- Validation Summary ---")
    for key, value in report.items():
        logger.info(f"  {key}: {value}")
    
    # Show problematic files
    problematic = validation_df[validation_df['status'] != 'OK']
    if len(problematic) > 0:
        logger.warning(f"\n{len(problematic)} files have issues:")
        logger.warning("\nTop 10 problematic files:")
        print(problematic.nlargest(10, 'num_issues')[
            ['filename', 'status', 'num_issues', 'num_lines']
        ].to_string())
    else:
        logger.info("\n[OK] All files passed validation!")
    
    # Analyze alignment quality
    aligned_data_file = config.ALIGNED_DATA_DIR / "aligned_data.parquet"
    if aligned_data_file.exists():
        logger.info("\n--- Analyzing Alignment Quality ---")
        analyzer = AlignmentAnalyzer(aligned_data_file)
        
        drift_stats = analyzer.analyze_drift_patterns()
        logger.info("\nDrift Statistics:")
        for key, value in drift_stats.items():
            logger.info(f"  {key}: {value:.2f} seconds")
        
        conf_stats = analyzer.analyze_confidence_distribution()
        logger.info("\nConfidence Statistics:")
        for key, value in conf_stats.items():
            logger.info(f"  {key}: {value:.2f}")
        
        # Create plots
        if config.CREATE_PLOTS:
            logger.info("\n--- Creating Diagnostic Plots ---")
            plot_dir = config.VALIDATION_DIR / "plots"
            analyzer.create_diagnostic_plots(plot_dir)
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION AND ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
