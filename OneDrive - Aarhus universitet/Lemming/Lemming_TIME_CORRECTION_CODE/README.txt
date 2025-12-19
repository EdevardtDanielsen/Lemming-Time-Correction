# Note for myself: D2 doesnt have may. maybe time_drift

RTC-TTN TIME ALIGNMENT PIPELINE
================================

Created by: Edevardt Johan Danielsen (ED)
Last edited: 18/11/2025

PURPOSE
-------
Correct greenhouse gas SD card data by aligning RTC timestamps to TTN ground truth, reconstruct clean hourly TXT files, and record validation/quality so each run is auditable per device.

OUTCOMES (WHAT YOU GET)
=======================
- Corrected hourly TXT files with dual timestamps (Original_RTC + Corrected_Time) at device sampling interval (config.DEVICE_SAMPLE_INTERVAL_SECONDS)
- Drift and confidence metrics per segment and per run, including anchor coverage by concat position
- Excel logbook entries (all-devices summary + per-device sheet) with confidence counts and issues
- Validation artifacts: direct TTN accuracy, 20% TTN holdout cross-validation, output QA, JSON summary
- Recovery of concatenated SD rows (missing newlines) with statistics and before/after integrity

RUN FAST
========
1) Set DEVICE_ID in config.py
2) Run main pipeline: python main_pipeline.py
3) (Optional) Deep validation: python True_Validation_Time_Correction.py
4) Outputs land in RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\ and TIME_CORRECTION_LOGBOOK\correction_log.xlsx

BATCH / OVERNIGHT RUN (ALL DEVICES)
===================================
Use `run_all_devices_overnight.ps1` to process a list of devices sequentially (pipeline + validation).
- Update the device list: open run_all_devices_overnight.ps1 and edit `$devices = @("A1", "A2", ...)` to match what you need.
- Check the virtual env path: script activates `C:\Users\au784422\Documents\Shared_Env\.env\Scripts\Activate.ps1`; adjust if your env differs.
- Run from this folder (or adjust Set-Location inside the script):
  powershell -ExecutionPolicy Bypass -File run_all_devices_overnight.ps1
- What it does per device: set DEVICE_ID in config.py → run `python main_pipeline.py` → if ok, run `python True_Validation_Time_Correction.py`.
- What you get: corrected files in RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\, validations in ...\validation\, Excel logbook updated for each device, terminal summary of successes/failures.

HOW THE PIPELINE WORKS (main_pipeline.py)
========================================
- Phase 1: Load + validate
  * SD loader (recursive) with concatenation recovery; filters RTC data before 2025-03-25 UTC
  * TTN loader (old + new formats); overlap check vs SD; optional validation JSON
- Phase 2: Segment
  * Split SD data on large gaps or backward jumps; records segment stats
- Phase 3: Align to TTN
  * Adaptive matching per segment using concat_position hints when present
  * Computes confidence and drift; logs anchor counts by concat position
- Phase 3.4: Backfill + group corrections
  * Propagate drift backwards into RTC-only regions
  * Group anchor correction: within each concat_group_id, propagate best drift so siblings move together
- Phase 3.5: Cross-check accuracy
  * Direct validation: nearest match to TTN within 30 s; MAE/median/RMSE/95th/max by confidence and concat_position
  * Cross-validation: hide 20% of TTN (if >=50 points), predict from aligned data, report same stats; optional CSV saved
- Phase 4: Reconstruct hourly files
  * Spread duplicate timestamps to 2 s spacing (uses device interval or RTC step)
  * Dynamic headers based on detected column count; write YYMMDDHH.TXT under month folders with dual timestamps
- Phase 5: Validate output files
  * Reads written files to check chronology, reversals, gaps, hour placement, mean drift per file; saves output_validation.csv
- Phase 6: Summarize + log
  * summary_report.json with alignment/validation stats; Excel logbook updated with drift, confidence splits, file counts, issues

CONCATENATION RECOVERY (DETAIL)
===============================
- Why: SD card write glitches can join multiple measurements on one line (missing newlines).
- Detection: count DD.MM.YYYY HH:MM:SS patterns per line; if >1, treat as concatenated and split.
- Types tracked: Type1 different timestamps; Type2 same timestamp different values; Type3 identical duplicates. Stats logged.
- Metadata: recovered rows carry concat_group_id (original physical line) and concat_position (ordering inside that line).
- Alignment impact: concat_position is passed to aligner for anchor diagnostics; group anchor correction later forces all rows in a concat_group_id to share the best drift anchor.
- Duplicate spreading: after alignment, any timestamp collisions are spread to preserve order before writing hourly files.
- Before/after example
  * Raw corrupted line (one physical line):
    "12.3,45.6,1000,23.1,45,25.03.2025 10:00:00 12.4,45.7,1001,23.2,46,25.03.2025 10:00:02"
  * Recovered rows (sorted):
    - concat_position=0, rtc=2025-03-25 10:00:00, data=[12.3,45.6,1000,23.1,45]
    - concat_position=1, rtc=2025-03-25 10:00:02, data=[12.4,45.7,1001,23.2,46]
  * Final hourly files: both appear in the 10:00 hour bucket with dual timestamps and even spacing, drift corrected relative to TTN anchors.

FILE-BY-FILE RESPONSIBILITIES
==============================
- config.py: Central device switch and all paths, formats, thresholds, toggles (logging, intermediate saves)
- data_loading.py: SD loader with concatenation recovery, validation of ranges/overlap; dual-format TTN loader (old headered + new server); gap/duplicate checks
- time_alignment.py: Segment alignment to TTN, confidence scoring, drift calculation, statistics helpers
- main_pipeline.py: Orchestrates phases; backward drift propagation; group anchor correction; reconstruction; validations; JSON + Excel reporting
- excel_logger.py: Writes per-run metrics and confidence counts to TIME_CORRECTION_LOGBOOK\correction_log.xlsx (summary + per-device sheet)
- ultimate_validation.py: Optional deep QA of corrected outputs (chronology, reversals, hour placement)
- True_Validation_Time_Correction.py: Entry point for ultimate_validation
- inspect_all_highest_segment.py: Diagnostic helper to inspect highest-confidence segments (use for troubleshooting alignment anchors)
- utils.py: Shared helpers

VALIDATION AND CROSS-CHECKING (WHERE RESULTS GO)
===============================================
- Direct validation: nearest TTN match stats; saved in validation_summary (and optionally CSV if enabled) under RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\validation\
- Cross-validation: 20% TTN holdout stats (MAE, RMSE, 95th, by predicted confidence); optional cross_validation_results.csv in validation folder
- Output QA: output_validation.csv checking chronology, reversals, gaps, hour placement, mean drift per file (validation folder)
- Summary: summary_report.json in OUTPUT_DIR; Excel logbook update with drift/confidence splits, file counts, and any issues

INPUTS AND OUTPUTS
==================
Inputs
- SD data: Lemming\Lemming 2025\data\{DEVICE_ID}\ (any subfolder depth)
- TTN reference: Lemming\TTN_TIME\{DEVICE_ID}\ (OLD and NEW formats auto-detected)

Outputs
- Corrected hourly files (dual timestamps): RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\{MM}\YYMMDDHH.TXT
- Logs: RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\logs\
- Validation: RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\validation\ (direct/cross-validation summaries, output QA, optional ultimate validation)
- Excel logbook: TIME_CORRECTION_LOGBOOK\correction_log.xlsx
- Intermediate (if enabled): aligned_data.parquet, alignment_statistics.csv, segment_summary.csv, cross_validation_results.csv, output_validation.csv, summary_report.json

DATA FORMATS
============
- SD input line: CH4_1,CH4_2,CO2,RH,Temp,Pressure,DD.MM.YYYY HH:MM:SS (recursive file search)
- TTN old format: Tab-separated with header (received_at, decoded_payload{unix})
- TTN new format: Space-separated, no header; supports single-line and two-line unix variants
- Corrected output line (example):
  Original_RTC,Sensor_Volt,CH4_Raw,CO2_ppm,Temp_C,RH_pct,Corrected_Time
  25.03.2025 10:00:00,12.3,45.6,1000,23.1,45,25.03.2025 10:00:00

DEPENDENCIES
============
Python packages: pandas, numpy, openpyxl, tqdm, python-dateutil, scikit-learn
Install: pip install pandas numpy openpyxl tqdm python-dateutil scikit-learn

TROUBLESHOOTING
===============
- "No SD card data could be loaded": ensure Lemming 2025\data\{DEVICE_ID}\ has .TXT files
- "TTN directory not found": ensure Lemming\TTN_TIME\{DEVICE_ID}\ exists
- Large errors or low confidence: check TTN coverage and SD/TTN overlap; review concatenation recovery stats in logs; inspect highest-confidence segments
- Excel logging failed: confirm openpyxl is installed and correction_log.xlsx is not open

CHANGING DEVICES
================
Edit config.py, set DEVICE_ID (e.g., "C3"), save, then run python main_pipeline.py. Paths update automatically.
