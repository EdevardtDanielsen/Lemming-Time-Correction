# =============================================================================
# Time Alignment Utilities
# =============================================================================
# Core functions for matching RTC times to TTN timestamps.
# Handles matching of drifting RTC clocks to accurate TTN reference times.
#
# Why this exists: Device RTC clocks drift over time. This module provides
# algorithms to align RTC timestamps with TTN ground truth, supporting
# multiple alignment strategies (anchor, interpolation, linear drift).
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# TTNTimeMatcher: Match RTC timestamps to TTN reference times
# Provides fast lookup and interpolation for finding closest TTN matches.
# Why: TTN timestamps are the ground truth for correcting RTC drift.
# -----------------------------------------------------------------------------
class TTNTimeMatcher:

    # Initialize with TTN reference data (DataFrame with 'unix' and 'received_at' columns)
    def __init__(self, ttn_df: pd.DataFrame):
        self.ttn_df = ttn_df.sort_values('unix').reset_index(drop=True)
        self.ttn_df['received_at'] = pd.to_datetime(self.ttn_df['received_at'], utc=True)
        
        # Create fast lookup structures
        self.unix_times = self.ttn_df['unix'].values
        self.received_times = self.ttn_df['received_at'].values
        
        logger.info(f"Initialized TTN matcher with {len(self.ttn_df)} reference points")
        logger.info(f"TTN time range: {self.received_times[0]} to {self.received_times[-1]}")
    
    # Find closest TTN timestamp to a given unix time
    # Returns dict with matched_unix, received_at, delta_seconds, match_quality
    # If no match within max_delta_seconds (default 1 day), tries interpolation
    def find_closest_match(self, unix_timestamp: int,
                          max_delta_seconds: int = 86400) -> Dict:
        if len(self.unix_times) == 0:
            return {
                'matched_unix': None,
                'received_at': None,
                'delta_seconds': None,
                'match_quality': 'no_reference_data'
            }
        
        # Find closest unix time
        deltas = np.abs(self.unix_times - unix_timestamp)
        closest_idx = np.argmin(deltas)
        delta = int(deltas[closest_idx])
        
        if delta <= max_delta_seconds:
            return {
                'matched_unix': int(self.unix_times[closest_idx]),
                'received_at': pd.Timestamp(self.received_times[closest_idx]),
                'delta_seconds': delta,
                'match_quality': self._assess_match_quality(delta)
            }
        else:
            # Try to interpolate from nearby matches
            return self._interpolate_from_neighbors(unix_timestamp, closest_idx, delta)
    
    # Assess quality of time match based on delta (excellent/good/acceptable/poor)
    def _assess_match_quality(self, delta_seconds: int) -> str:
        if delta_seconds <= 10:
            return 'excellent'
        elif delta_seconds <= 300:  # was 60s; relax to 5 minutes
            return 'good'
        elif delta_seconds <= 3600:
            return 'acceptable'
        else:
            return 'poor'
    
    # Interpolate timestamp from nearby matches when no close match exists
    # Looks for TTN anchors within +/- 7 days and calculates average drift
    def _interpolate_from_neighbors(self, unix_timestamp: int,
                                   closest_idx: int,
                                   delta: int) -> Dict:
        # Look for matches within +/- 7 days
        search_window = 86400 * 7
        mask = np.abs(self.unix_times - unix_timestamp) <= search_window
        
        if not mask.any():
            return {
                'matched_unix': None,
                'received_at': None,
                'delta_seconds': delta,
                'match_quality': 'no_nearby_reference'
            }
        
        # Calculate average drift from nearby points
        nearby_unix = self.unix_times[mask]
        nearby_received = self.received_times[mask]
        
        # Convert to timestamps for calculation - ensure both are UTC aware
        unix_as_datetime = pd.to_datetime(nearby_unix, unit='s', utc=True)
        received_as_datetime = pd.to_datetime(nearby_received, utc=True)

        # Calculate drift (how much RTC is behind/ahead of reality)
        drifts = (received_as_datetime - unix_as_datetime).total_seconds()
        avg_drift = np.mean(drifts)
        
        # Apply average drift to estimate true time
        estimated_time = pd.to_datetime(unix_timestamp, unit='s', utc=True) + \
                        pd.Timedelta(seconds=avg_drift)
        
        return {
            'matched_unix': None,
            'received_at': estimated_time,
            'delta_seconds': delta,
            'match_quality': 'interpolated',
            'interpolation_sample_size': int(mask.sum()),
            'avg_drift_seconds': float(avg_drift)
        }
    
    # Estimate drift rate (seconds per second) between two unix timestamps
    # Returns None if cannot be estimated (missing matches or zero duration)
    def estimate_drift_rate(self, start_unix: int, end_unix: int) -> Optional[float]:
        if len(self.unix_times) < 2:
            return None
        
        # Find matches near start and end
        start_match = self.find_closest_match(start_unix)
        end_match = self.find_closest_match(end_unix)
        
        if (start_match['received_at'] is None or 
            end_match['received_at'] is None):
            return None
        
        # Calculate drift
        rtc_duration = end_unix - start_unix
        ttn_duration = (end_match['received_at'] - 
                       start_match['received_at']).total_seconds()
        
        if rtc_duration == 0:
            return None
        
        drift_rate = (ttn_duration - rtc_duration) / rtc_duration
        return drift_rate


# -----------------------------------------------------------------------------
# SegmentTimeAligner: Align timestamps within a segment using various strategies
# Supports: adaptive, anchor, interpolate, and linear_drift alignment methods.
# Why: Different data characteristics require different alignment approaches.
# -----------------------------------------------------------------------------
class SegmentTimeAligner:

    def __init__(self, ttn_matcher: TTNTimeMatcher):
        self.ttn_matcher = ttn_matcher
    
    # Align a segment of RTC times to TTN times
    # strategy: 'adaptive' (auto-select), 'anchor', 'interpolate', or 'linear_drift'
    # positions: Optional concat_position series for per-position drift handling
    # Returns DataFrame with aligned times and quality metrics
    def align_segment(self, rtc_times: pd.Series,
                     strategy: str = 'adaptive',
                     positions: Optional[pd.Series] = None) -> pd.DataFrame:
        # If per-position information is provided, align each position separately
        if positions is not None:
            return self._align_with_positions(rtc_times, positions, strategy)

        unix_times = rtc_times.apply(lambda x: int(x.timestamp())).values
        
        if strategy == 'adaptive':
            return self._adaptive_alignment(rtc_times, unix_times)
        elif strategy == 'anchor':
            return self._anchor_alignment(rtc_times, unix_times)
        elif strategy == 'interpolate':
            return self._interpolate_alignment(rtc_times, unix_times)
        elif strategy == 'linear_drift':
            return self._linear_drift_alignment(rtc_times, unix_times)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    # Align timestamps separately for each concat_position to capture position-specific drift
    # Falls back to global alignment if only one position exists
    # If a position has no usable anchors, borrows drift from the anchored position
    def _align_with_positions(self, rtc_times: pd.Series, positions: pd.Series, strategy: str) -> pd.DataFrame:
        unique_positions = sorted(positions.dropna().unique().tolist())
        if len(unique_positions) <= 1:
            return self.align_segment(rtc_times, strategy=strategy, positions=None)

        aligned_parts = []

        # Choose anchor position (prefer 0)
        anchor_pos = 0 if 0 in unique_positions else unique_positions[0]
        anchor_mask = positions == anchor_pos
        anchor_times = rtc_times[anchor_mask]

        anchor_aligned = self.align_segment(anchor_times, strategy=strategy, positions=None)
        anchor_aligned['concat_position'] = anchor_pos
        anchor_aligned['__orig_idx'] = anchor_times.index

        # Base drift from anchor (median)
        try:
            base_drift = (pd.to_datetime(anchor_aligned['aligned_time'], utc=True) -
                          pd.to_datetime(anchor_aligned['original_rtc'], utc=True)).median()
        except Exception:
            base_drift = None

        aligned_parts.append(anchor_aligned)

        for pos in unique_positions:
            if pos == anchor_pos:
                continue

            pos_mask = positions == pos
            pos_times = rtc_times[pos_mask]

            # Try normal alignment
            pos_aligned = self.align_segment(pos_times, strategy=strategy, positions=None)

            # Determine if this position has any decent anchors
            has_anchor = 'confidence' in pos_aligned.columns and (pos_aligned['confidence'] >= 0.7).any()

            if not has_anchor and base_drift is not None:
                # Fallback: reuse anchor drift
                shifted_times = pd.to_datetime(pos_aligned['original_rtc'], utc=True) + base_drift
                pos_aligned['aligned_time'] = shifted_times
                pos_aligned['confidence'] = 0.65  # mid confidence since it's borrowed
                pos_aligned['method'] = 'fallback_anchor_drift'

            pos_aligned['concat_position'] = pos
            pos_aligned['__orig_idx'] = pos_times.index
            aligned_parts.append(pos_aligned)

        # Combine and restore original order
        combined = pd.concat(aligned_parts, ignore_index=True)
        combined = combined.set_index('__orig_idx').sort_index()
        combined = combined.reset_index(drop=True)
        return combined
    
    # Adaptively choose best alignment strategy based on data characteristics
    # Decision logic:
    #   - Multiple anchors in range (>=3) -> use interpolation (piecewise-linear)
    #   - Few anchors but good endpoints -> use linear drift
    #   - One good anchor -> use anchor alignment
    #   - No good matches -> interpolation will extrapolate from nearest
    # Why piecewise > linear: np.interp() uses ALL anchors, not just endpoints,
    # which is more accurate for segments spanning days/weeks/months.
    def _adaptive_alignment(self, rtc_times: pd.Series,
                           unix_times: np.ndarray) -> pd.DataFrame:
        # Count TTN anchors within the segment's time range (with some padding)
        segment_start = unix_times[0]
        segment_end = unix_times[-1]
        padding = 3600  # 1 hour padding on each side

        anchors_in_range = np.sum(
            (self.ttn_matcher.unix_times >= segment_start - padding) &
            (self.ttn_matcher.unix_times <= segment_end + padding)
        )

        # Also check endpoint match quality for fallback decisions
        first_match = self.ttn_matcher.find_closest_match(unix_times[0])
        last_match = self.ttn_matcher.find_closest_match(unix_times[-1])

        good_matches = sum([
            first_match['match_quality'] in ['excellent', 'good', 'acceptable'],
            last_match['match_quality'] in ['excellent', 'good', 'acceptable']
        ])

        # DECISION LOGIC:
        # 1. If we have multiple anchors in range -> use interpolation (piecewise-linear)
        # 2. If we have few anchors but good endpoints -> use linear drift
        # 3. If we have only one good anchor -> use anchor alignment
        # 4. Otherwise -> interpolation will extrapolate from nearest anchors

        if anchors_in_range >= 3:
            # Multiple anchors available - use piecewise interpolation for best accuracy
            logger.debug(f"Using interpolation ({anchors_in_range} TTN anchors in segment range)")
            return self._interpolate_alignment(rtc_times, unix_times)
        elif good_matches >= 2:
            # Few anchors but good endpoints - linear drift is acceptable
            logger.debug("Using linear drift model (few anchors, good endpoints)")
            return self._linear_drift_alignment(rtc_times, unix_times)
        elif good_matches >= 1:
            # Have one good anchor - use it as reference
            logger.debug("Using anchor alignment (one good match)")
            return self._anchor_alignment(rtc_times, unix_times)
        else:
            # No good matches - interpolation will extrapolate from nearest
            logger.debug("Using interpolation (no good matches, will extrapolate)")
            return self._interpolate_alignment(rtc_times, unix_times)
    
    # Align using first timestamp as anchor, then apply RTC intervals
    # Uses first TTN match as reference point and preserves RTC time deltas
    def _anchor_alignment(self, rtc_times: pd.Series,
                         unix_times: np.ndarray) -> pd.DataFrame:
        results = []
        
        # Get anchor match
        anchor_match = self.ttn_matcher.find_closest_match(unix_times[0])
        
        if anchor_match['received_at'] is None:
            # Fallback to direct matching for each point
            return self._interpolate_alignment(rtc_times, unix_times)
        
        anchor_time = anchor_match['received_at']
        anchor_rtc = rtc_times.iloc[0]
        
        # Apply RTC intervals from anchor
        for i, (rtc_time, unix_time) in enumerate(zip(rtc_times, unix_times)):
            rtc_delta = rtc_time - anchor_rtc
            aligned_time = anchor_time + rtc_delta
            
            results.append({
                'original_rtc': rtc_time,
                'unix_time': unix_time,
                'aligned_time': aligned_time,
                'confidence': self._calculate_confidence(anchor_match['match_quality']),
                'method': 'anchor',
                'reference_delta_seconds': anchor_match['delta_seconds']
            })
        
        return pd.DataFrame(results)
        
    # Interpolate drift from TTN anchors and apply to each timestamp
    # Uses piecewise-linear interpolation between all TTN anchor points
    # Confidence scoring distinguishes interpolation (high) vs extrapolation (lower)
    def _interpolate_alignment(self, rtc_times: pd.Series,
                                unix_times: np.ndarray) -> pd.DataFrame:
            # Get TTN reference times and calculate drift
            ttn_unix = self.ttn_matcher.unix_times
            ttn_received = self.ttn_matcher.received_times
            
            # Convert to seconds for interpolation
            ttn_unix_sec = ttn_unix.astype('float64')
            ttn_received_sec = pd.to_datetime(ttn_received, utc=True).astype('int64') / 1e9
            
            # Calculate drift at each TTN anchor (how much RTC is behind/ahead)
            ttn_drift = ttn_received_sec - ttn_unix_sec
            
            # Interpolate drift for our RTC times
            unix_times_float = unix_times.astype('float64')
            interpolated_drift = np.interp(unix_times_float, ttn_unix_sec, ttn_drift)
            
            # Get anchor range boundaries for interpolation vs extrapolation detection
            anchor_min = ttn_unix_sec.min()
            anchor_max = ttn_unix_sec.max()

            # Apply drift to get corrected times
            results = []
            for i, (rtc_time, unix_time, drift) in enumerate(zip(rtc_times, unix_times, interpolated_drift)):
                aligned_time = pd.to_datetime(unix_time + drift, unit='s', utc=True)

                # IMPROVED CONFIDENCE SCORING (Dec 2025):
                # Distinguish between interpolation (between anchors) and extrapolation (beyond)

                if unix_time < anchor_min:
                    # EXTRAPOLATING before first anchor
                    distance_from_boundary = anchor_min - unix_time
                    if distance_from_boundary < 3600:      # < 1 hour from boundary
                        confidence = 0.5
                    elif distance_from_boundary < 86400:   # < 1 day
                        confidence = 0.3
                    else:
                        confidence = 0.2

                elif unix_time > anchor_max:
                    # EXTRAPOLATING after last anchor
                    distance_from_boundary = unix_time - anchor_max
                    if distance_from_boundary < 3600:
                        confidence = 0.5
                    elif distance_from_boundary < 86400:
                        confidence = 0.3
                    else:
                        confidence = 0.2

                else:
                    # INTERPOLATING between anchors - find surrounding anchors
                    # Find indices of anchors before and after this point
                    idx_after = np.searchsorted(ttn_unix_sec, unix_time)
                    idx_before = idx_after - 1

                    if idx_before >= 0 and idx_after < len(ttn_unix_sec):
                        # Calculate gap between surrounding anchors
                        anchor_gap = ttn_unix_sec[idx_after] - ttn_unix_sec[idx_before]

                        if anchor_gap < 3600:         # Anchors < 1 hour apart
                            confidence = 0.9
                        elif anchor_gap < 7200:       # < 2 hours
                            confidence = 0.85
                        elif anchor_gap < 14400:      # < 4 hours
                            confidence = 0.8
                        elif anchor_gap < 86400:      # < 1 day
                            confidence = 0.7
                        else:
                            confidence = 0.5
                    else:
                        # Edge case - use distance-based fallback
                        nearest_ttn_distance = np.min(np.abs(ttn_unix_sec - unix_time))
                        confidence = 0.7 if nearest_ttn_distance < 3600 else 0.5

                results.append({
                    'original_rtc': rtc_time,
                    'unix_time': unix_time,
                    'aligned_time': aligned_time,
                    'confidence': confidence,
                    'method': 'interpolated_drift',
                    'drift_seconds': float(drift)
                })
            
            return pd.DataFrame(results)
    
    # Convert match quality string to confidence score (0.0 to 1.0)
    def _calculate_confidence(self, match_quality: str) -> float:
        quality_map = {
            'excellent': 1.0,
            'good': 0.9,
            'acceptable': 0.7,
            'poor': 0.5,
            'interpolated': 0.7,
            'no_nearby_reference': 0.3,
            'no_reference_data': 0.1
        }
        return quality_map.get(match_quality, 0.5)

    # Model linear drift between first and last timestamps
    # Calculates drift rate from endpoints and applies uniform correction
    # Best for short segments with good endpoint matches
    def _linear_drift_alignment(self, rtc_times: pd.Series,
                                unix_times: np.ndarray) -> pd.DataFrame:
        results = []
        
        # Get matches for endpoints
        first_match = self.ttn_matcher.find_closest_match(unix_times[0])
        last_match = self.ttn_matcher.find_closest_match(unix_times[-1])
        
        if (first_match['received_at'] is None or 
            last_match['received_at'] is None):
            # Fallback
            return self._anchor_alignment(rtc_times, unix_times)
        
        # Calculate drift model - ENSURE ALL ARE TIMEZONE-AWARE
        t0_rtc = pd.to_datetime(rtc_times.iloc[0], utc=True)
        t0_ttn = pd.to_datetime(first_match['received_at'], utc=True)
        t1_rtc = pd.to_datetime(rtc_times.iloc[-1], utc=True)
        t1_ttn = pd.to_datetime(last_match['received_at'], utc=True)
        
        rtc_duration = (t1_rtc - t0_rtc).total_seconds()
        ttn_duration = (t1_ttn - t0_ttn).total_seconds()
        
        if rtc_duration == 0:
            return self._anchor_alignment(rtc_times, unix_times)
        
        drift_rate = (ttn_duration - rtc_duration) / rtc_duration
        
        # Apply drift model to each timestamp
        for i, (rtc_time, unix_time) in enumerate(zip(rtc_times, unix_times)):
            # Ensure rtc_time is timezone-aware
            rtc_time_utc = pd.to_datetime(rtc_time, utc=True)
            
            rtc_elapsed = (rtc_time_utc - t0_rtc).total_seconds()
            drift_correction = timedelta(seconds=rtc_elapsed * drift_rate)
            aligned_time = rtc_time_utc + drift_correction + (t0_ttn - t0_rtc)
            
            results.append({
                'original_rtc': rtc_time,
                'unix_time': unix_time,
                'aligned_time': aligned_time,
                'confidence': 0.85,
                'method': 'linear_drift',
                'drift_rate': drift_rate
            })
        
        return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# compute_alignment_statistics: Compute statistics on alignment quality
# Returns dict with drift stats (mean, median, std, min, max), confidence
# metrics (mean, % high/low), and alignment method breakdown.
# Why: Provides quantitative assessment of how well the alignment worked.
# -----------------------------------------------------------------------------
def compute_alignment_statistics(aligned_df: pd.DataFrame) -> Dict:
    if len(aligned_df) == 0:
        return {}
    
    # Calculate drift simply - iterate if needed to avoid array type issues
    drifts = []
    for i in range(len(aligned_df)):
        try:
            aligned_t = aligned_df['aligned_time'].iloc[i]
            original_t = aligned_df['original_rtc'].iloc[i]
            
            # Ensure both are pandas Timestamps with timezone
            if not isinstance(aligned_t, pd.Timestamp):
                aligned_t = pd.Timestamp(aligned_t)
            if not isinstance(original_t, pd.Timestamp):
                original_t = pd.Timestamp(original_t)
            
            # Make sure both have UTC timezone
            if aligned_t.tz is None:
                aligned_t = aligned_t.tz_localize('UTC')
            if original_t.tz is None:
                original_t = original_t.tz_localize('UTC')
            
            drift = (aligned_t - original_t).total_seconds()
            drifts.append(drift)
        except Exception as e:
            logger.debug(f"Could not calculate drift for row {i}: {e}")
            drifts.append(0)
    
    drift_series = pd.Series(drifts)
    
    stats = {
        'num_samples': len(aligned_df),
        'mean_drift_seconds': float(drift_series.mean()),
        'median_drift_seconds': float(drift_series.median()),
        'std_drift_seconds': float(drift_series.std()),
        'max_drift_seconds': float(drift_series.max()),
        'min_drift_seconds': float(drift_series.min()),
        'mean_confidence': float(aligned_df['confidence'].mean()),
        'pct_high_confidence': float((aligned_df['confidence'] >= 0.9).mean() * 100),
        'pct_low_confidence': float((aligned_df['confidence'] < 0.6).mean() * 100),
        'alignment_methods': aligned_df['method'].value_counts().to_dict() if 'method' in aligned_df.columns else {}
    }
    
    return stats
