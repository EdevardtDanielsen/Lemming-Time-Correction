# =============================================================================
# Configuration for RTC-TTN Time Alignment Pipeline
# =============================================================================
# UNIVERSAL VERSION - Works for any device (H1, C3, A1, B1, etc.)
# Change DEVICE_ID below to process different devices.
#
# Created by: Edevardt Johan Danielsen
# =============================================================================
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ============================================================================
# DEVICE CONFIGURATION - CHANGE THIS FOR DIFFERENT DEVICES!
# ============================================================================
DEVICE_ID = "B1"   # "H1", "C3", "A1", "B1", etc.

# ============================================================================
# DEVICE-SPECIFIC SAMPLING INTERVALS (from diagnostic analysis)
# ============================================================================
DEVICE_BASE_DT = {
    'A1': 6, 'A2': 5, 'A3': 5,
    'B1': 5, 'B2': 5, 'B3': 6,
    'C1': 6, 'C2': 5, 'C3': 6,
    'D1': 5, 'D2': 5, 'D3': 5,
    'E1': 6, 'E2': 5, 'E3': 4,
    'F1': 5, 'F2': 5, 'F3': 6,
    'G1': 5, 'G2': 6, 'G3': 5,
    'H1': 5, 'H2': 12, 'H3': 5
}

# Get current device's sampling interval
DEVICE_SAMPLE_INTERVAL_SECONDS = DEVICE_BASE_DT.get(DEVICE_ID, 5)  # Default to 5 if unknown

# ============================================================================
# BASE DIRECTORY CONFIGURATION
# ============================================================================
BASE_DIR = Path(r"C:\Users\au784422\OneDrive - Aarhus universitet\Lemming")

# ============================================================================
# AUTO-GENERATED PATHS (Based on DEVICE_ID)
# ============================================================================
# Input paths
SD_CARD_DATA_DIR = BASE_DIR / "Lemming 2025" / "data" / DEVICE_ID
TTN_DATA_DIR = BASE_DIR / "TTN_TIME" / DEVICE_ID  # Folder with OLD + NEW files

# Output paths
OUTPUT_DIR = BASE_DIR / "RAW_DATA_TIME_CORRECTED" / DEVICE_ID
ALIGNED_DATA_DIR = OUTPUT_DIR / "aligned_segments"
RECONSTRUCTED_DATA_DIR = OUTPUT_DIR  # Files go directly in device folder (with month subfolders)
VALIDATION_DIR = OUTPUT_DIR / "validation"
LOGS_DIR = OUTPUT_DIR / "logs"

# Excel logging path
LOGBOOK_DIR = BASE_DIR / "TIME_CORRECTION_LOGBOOK"
LOGBOOK_FILE = LOGBOOK_DIR / "correction_log.xlsx"

# ============================================================================
# TIME ALIGNMENT PARAMETERS
# ============================================================================
# Valid RTC range - data outside this is likely corrupted
RTC_VALID_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
RTC_VALID_END = datetime(2026, 12, 31, tzinfo=timezone.utc)

# ============================================================================
# RTC TIMEZONE OFFSET (CRITICAL FOR CORRECT TIME INTERPRETATION)
# ============================================================================
# The Arduino RTC was programmed using __TIME__ macro at compile time.
# This captures the LOCAL time of the computer that compiled the firmware.
# The RTC has NO DST awareness - it stores a FIXED offset from UTC.
#
# If compiled in Copenhagen during winter (Nov-Mar): UTC+1 (offset = 60 minutes)
# If compiled in Copenhagen during summer (Mar-Oct): UTC+2 (offset = 120 minutes)
#
# IMPORTANT: Do NOT use pytz.timezone('Europe/Copenhagen') as it applies DST!
# Use a fixed offset based on when the device was actually compiled.
#
# Device-specific offsets (minutes from UTC):
DEVICE_RTC_OFFSET_MINUTES = {
    'A1': 60,   # Compiled in winter (UTC+1)
    'A2': 60,   # Compiled in winter (UTC+1) - verify!
    'A3': 60,   # Compiled in winter (UTC+1) - verify!
    'B1': 60,   # Compiled in winter (UTC+1) - verify!
    'B2': 60,   # Compiled in winter (UTC+1) - verify!
    'B3': 60,   # Default assumption
    'C1': 60,   # Default assumption
    'C2': 60,   # Default assumption
    'C3': 60,   # Default assumption
    'D1': 60,   # Default assumption
    'D2': 60,   # Default assumption
    'D3': 60,   # Default assumption
    'E1': 60,   # Default assumption
    'E2': 60,   # Default assumption
    'E3': 60,   # Default assumption
    'F1': 60,   # Default assumption
    'F2': 60,   # Default assumption
    'F3': 60,   # Default assumption
    'G1': 60,   # Default assumption
    'G2': 60,   # Default assumption
    'G3': 60,   # Default assumption
    'H1': 60,   # Default assumption
    'H2': 60,   # Default assumption
    'H3': 60,   # Default assumption
}

# Get current device's RTC offset
DEVICE_RTC_OFFSET = DEVICE_RTC_OFFSET_MINUTES.get(DEVICE_ID, 60)  # Default to UTC+1

# Maximum allowed gap between consecutive readings before starting new segment
MAX_INTRA_SEGMENT_GAP = timedelta(minutes=2)

# Maximum allowed deviation between RTC and TTN for a "good" match (1 day)
MAX_MATCH_DELTA = timedelta(days=1)

# When interpolating within segments, expected sampling interval (device-specific)
EXPECTED_SAMPLE_INTERVAL = timedelta(seconds=DEVICE_SAMPLE_INTERVAL_SECONDS)

# Minimum segment size to process (lines)
MIN_SEGMENT_SIZE = 5

# ============================================================================
# DATA FORMAT PARAMETERS
# ============================================================================
# Expected format of RTC timestamp in SD card data
RTC_FORMAT = "%d.%m.%Y %H:%M:%S"

# TTN format detection patterns
TTN_OLD_FORMAT_INDICATOR = "received_at"  # Tab-separated with this header
TTN_NEW_FORMAT_PATTERN = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"  # Starts with timestamp

# Output timestamp format
OUTPUT_FORMAT = "%d.%m.%Y %H:%M:%S"

# Expected columns in SD card data (comma-separated)
EXPECTED_COLUMNS = ["CH4_1", "CH4_2", "CO2", "RH", "Temp", "Pressure", "RTC_timestamp"]

# ============================================================================
# QUALITY CONTROL THRESHOLDS
# ============================================================================
# Confidence score thresholds
CONFIDENCE_HIGH = 0.9      # Excellent match
CONFIDENCE_MEDIUM = 0.6    # Acceptable match
CONFIDENCE_LOW = 0.3       # Poor match (interpolated/fallback)

# File validation thresholds (device-specific)
# Calculate based on device sampling interval: 3600 seconds/hour ÷ sample_interval
SAMPLES_PER_HOUR = 3600 // DEVICE_SAMPLE_INTERVAL_SECONDS
MIN_EXPECTED_LINES_PER_HOUR = int(SAMPLES_PER_HOUR * 0.85)  # Allow 15% loss
MAX_EXPECTED_LINES_PER_HOUR = int(SAMPLES_PER_HOUR * 1.10)  # Allow 10% excess
MAX_ALLOWED_TIME_GAP_MINUTES = 5

# ============================================================================
# PROCESSING FLAGS
# ============================================================================
# Enable verbose logging
VERBOSE = True

# Save intermediate outputs for debugging
SAVE_INTERMEDIATE_FILES = True

# Perform extensive validation checks
RUN_VALIDATION = True

# Create diagnostic plots
CREATE_PLOTS = False  # Set to True if you want matplotlib plots

# Excel logging
ENABLE_EXCEL_LOGGING = True

# ============================================================================
# VALIDATION
# ============================================================================
# -----------------------------------------------------------------------------
# validate_config: Check that all required paths exist
# Returns lists of errors (blocking) and warnings (non-blocking).
# Why: Catches missing directories before pipeline runs to fail fast.
# -----------------------------------------------------------------------------
def validate_config():
    errors = []
    warnings = []

    # Check SD card data directory exists
    if not SD_CARD_DATA_DIR.exists():
        errors.append(f"SD card data directory not found: {SD_CARD_DATA_DIR}")

    # Check TTN data directory exists
    if not TTN_DATA_DIR.exists():
        errors.append(f"TTN data directory not found: {TTN_DATA_DIR}")

    # Check TTN data directory has files
    if TTN_DATA_DIR.exists():
        txt_files = list(TTN_DATA_DIR.glob("*.txt"))
        if len(txt_files) == 0:
            errors.append(f"No .txt files found in TTN directory: {TTN_DATA_DIR}")

    # Warnings for output directories (will be created)
    if not OUTPUT_DIR.exists():
        warnings.append(f"Output directory will be created: {OUTPUT_DIR}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Test configuration when run directly: python config.py
# Prints paths, validates directories, and shows file counts.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("CONFIGURATION TEST")
    print("=" * 80)
    print(f"\nDevice ID: {DEVICE_ID}")
    print(f"\nPaths:")
    print(f"  SD Card Data: {SD_CARD_DATA_DIR}")
    print(f"  TTN Data: {TTN_DATA_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Logbook: {LOGBOOK_FILE}")

    print(f"\nValidating configuration...")
    errors, warnings = validate_config()

    if errors:
        print(f"\n[ERROR] ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print(f"\n[WARNING]  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors:
        print(f"\n[OK] Configuration valid!")

        # Show what files would be loaded
        if SD_CARD_DATA_DIR.exists():
            sd_files = list(SD_CARD_DATA_DIR.rglob("*.TXT"))
            print(f"\nFound {len(sd_files)} SD card .TXT files")

        if TTN_DATA_DIR.exists():
            ttn_files = list(TTN_DATA_DIR.glob("*.txt"))
            print(f"Found {len(ttn_files)} TTN .txt files")
