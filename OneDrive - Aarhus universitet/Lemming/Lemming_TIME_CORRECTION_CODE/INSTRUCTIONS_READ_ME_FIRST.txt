================================================================================
                    LEMMING TIME CORRECTION CODE
                       Quick Start Instructions
================================================================================

Created by: Edevardt Johan Danielsen (ED)
Last updated: December 2025

--------------------------------------------------------------------------------
WHAT IS THIS PROJECT?
--------------------------------------------------------------------------------

This project fixes a common problem with field sensors: their internal clocks
drift over time. Our Lemming greenhouse gas sensors store measurements on SD
cards with timestamps from their internal clock (RTC). Unfortunately, these
clocks can drift by seconds per day - after months in the field, timestamps
could be off by many minutes.

The solution: When sensors send data via LoRaWAN (TTN), the network records
the exact time the message arrived. We use these accurate TTN timestamps as
"ground truth" to correct the drifting SD card timestamps.

In simple terms: We compare the sensor's clock to internet time, calculate
how far off it is, and fix all the SD card timestamps accordingly.


--------------------------------------------------------------------------------
HOW TO RUN IT
--------------------------------------------------------------------------------

Quick version:
   1. Open config.py and set DEVICE_ID = "H1" (or whichever device you want)
   2. Run: python main_pipeline.py
   3. Find corrected files in: RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\

For multiple devices overnight:
   Run: powershell -File run_all_devices_overnight.ps1
   (Edit the script first to list which devices you want to process)


--------------------------------------------------------------------------------
WHERE ARE MY FILES?
--------------------------------------------------------------------------------

INPUT (what you need):
   - SD card data:  Lemming\Lemming 2025\data\{DEVICE_ID}\
   - TTN reference: Lemming\TTN_TIME\{DEVICE_ID}\

OUTPUT (what you get):
   - Corrected hourly files: RAW_DATA_TIME_CORRECTED\{DEVICE_ID}\
   - Processing log:         TIME_CORRECTION_LOGBOOK\correction_log.xlsx


--------------------------------------------------------------------------------
WHY ARE THERE SO MANY FILES?
--------------------------------------------------------------------------------

Short answer: Each file has one specific job. This makes the code easier to
maintain and debug.

The files are organized like a production line:
   1. config.py sets up what device to process and where files are
   2. data_loading.py reads the raw data from SD cards and TTN
   3. time_alignment.py figures out how to correct the timestamps
   4. main_pipeline.py orchestrates everything in the right order
   5. Validation scripts check if the output looks correct
   6. Helper files (excel_logger.py, utils.py) provide supporting functions


--------------------------------------------------------------------------------
FILE-BY-FILE GUIDE
--------------------------------------------------------------------------------

MAIN FILES (you'll use these directly):

   config.py
      What: Settings file - change DEVICE_ID here to switch devices
      When to edit: Every time you process a new device

   main_pipeline.py
      What: The main program that does everything
      How to run: python main_pipeline.py
      Output: Corrected hourly TXT files, logs, validation reports

   run_all_devices_overnight.ps1
      What: PowerShell script to process multiple devices automatically
      How to run: powershell -File run_all_devices_overnight.ps1
      When to use: Processing several devices in one go (overnight)


BEHIND-THE-SCENES FILES (you don't run these directly):

   data_loading.py
      What: Reads SD card files and TTN reference data
      Special: Can recover data when SD card write errors caused lines to
               merge together (concatenation recovery)

   time_alignment.py
      What: The "brain" that figures out timestamp corrections
      How it works: Compares device clock to TTN clock, calculates drift,
                    interpolates corrections for all timestamps

   excel_logger.py
      What: Records each processing run in an Excel logbook
      Output: TIME_CORRECTION_LOGBOOK\correction_log.xlsx


VALIDATION FILES (check if output is correct):

   True_Validation_Time_Correction.py
      What: Detailed validation of output files
      When to use: After main pipeline, to double-check results
      How to run: python True_Validation_Time_Correction.py

   ultimate_validation.py
      What: Also does detailed validation (similar to above)
      Note: This is essentially the same as True_Validation_Time_Correction.py
            Both exist for historical reasons - you only need to run one

   validation_analysis.py
      What: Creates charts and analysis of alignment quality
      When to use: If you want visual diagnostics (plots)


UTILITY FILES:

   utils.py
      What: Helper functions used by other files
      Contains: Data loading helpers, quality report generators, etc.


DOCUMENTATION FILES (you're reading one now):

   INSTRUCTIONS_READ_ME_FIRST.txt  <-- This file (simple guide)
   README.txt                       <-- Detailed technical reference
   TIME_ALIGNMENT_DOCUMENTATION.txt <-- Deep dive into the alignment algorithm
   TIME_EXPLANATION.txt             <-- Explains the different time sources


--------------------------------------------------------------------------------
ABOUT THE DUPLICATE VALIDATION SCRIPTS
--------------------------------------------------------------------------------

You may notice two files that seem to do the same thing:
   - True_Validation_Time_Correction.py
   - ultimate_validation.py

Both perform comprehensive validation of the corrected output files. They
check for:
   - Timestamps in chronological order
   - No duplicate timestamps
   - No large gaps in the data
   - Correct hour assignments in filenames

Why two files? They evolved separately during development. The "ultimate"
version is slightly newer (November 21 vs November 18) and includes a few
extra features like importing pipeline validation results.

Which to use? Either works. The overnight script uses True_Validation.
You could safely delete one, but keeping both doesn't cause any problems.


--------------------------------------------------------------------------------
COMMON ISSUES
--------------------------------------------------------------------------------

"No SD card data could be loaded"
   --> Check that Lemming 2025\data\{DEVICE_ID}\ contains .TXT files

"TTN directory not found"
   --> Make sure Lemming\TTN_TIME\{DEVICE_ID}\ exists

"Excel logging failed"
   --> Close correction_log.xlsx if it's open in Excel
   --> Make sure openpyxl is installed: pip install openpyxl

Results look wrong (large time errors)
   --> Check if there's enough TTN coverage for your data period
   --> Look at the confidence scores in the log - low confidence = less reliable


--------------------------------------------------------------------------------
CHANGING DEVICES
--------------------------------------------------------------------------------

1. Open config.py
2. Find the line: DEVICE_ID = "H1"
3. Change "H1" to your device (e.g., "C3", "A1", "B1", etc.)
4. Save the file
5. Run: python main_pipeline.py

All paths update automatically based on DEVICE_ID.


================================================================================
                              END OF INSTRUCTIONS
================================================================================
