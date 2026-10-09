# OpenSky Data Audit Report

## 1. Dataset Overview
- **Total Rows**: 4,344,359
- **Total Columns**: 11
- **Duplicate Records**: 0

## 2. Identified Key Columns
- **Flight Identifier**: `callsign`
- **Timestamp**: `timestamp`
- **Altitude**: `altitude`
- **Velocity**: `groundspeed`
- **Heading**: `track`
- **Vertical Rate**: `vertical_rate`
- **Latitude**: `latitude`
- **Longitude**: `longitude`

## 3. Missing Values Summary (>0%)
- **groundspeed**: 37,047 missing (0.85%)
- **track**: 37,047 missing (0.85%)
- **vertical_rate**: 37,047 missing (0.85%)

## 4. Timestamp Range
- **Minimum Timestamp**: 1514809789000.0
- **Maximum Timestamp**: 1580270992000.0

## 5. Flight Trajectory & Sampling Stats
- **Unique Flights**: 823
- **Observations per Flight**:
  - Mean: 5278.69
  - Median: 4398.00
  - Min: 33
  - Max: 22,825

- **Sampling Intervals (Seconds)**:
  - Mean: 19764.13
  - Median: 1000.00
  - Min: 1000.00
  - Max: 17962957000.00

## 6. Sequence Window Viability (Min 15 obs)
- **Viable Flights**: 823
- **Insufficient Flights**: 0

## 7. Conclusions & Limitations
- All essential columns (identifier, time, position, altitude, velocity, heading, vertical rate) were successfully identified.
