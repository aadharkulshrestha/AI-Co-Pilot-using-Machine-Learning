# NASA ASRS & OpenSky Stage 1 Dataset Profiling Report

**Generated Date:** 2026-09-15  
**Pipeline Stage:** Stage 1 (Ingestion, Cleaning, Normalization & Target Derivation)

---

## 1. Executive Summary

| Metric | NASA ASRS Dataset | OpenSky Metadata |
| :--- | :--- | :--- |
| **Raw Uploaded Records** | 7,029 | 832 |
| **Deduplicated Records** | 6,978 (51 duplicates removed) | 832 (0 duplicates) |
| **Out-of-Window Records** | 1 (July 2012 legacy record removed) | 0 |
| **Final Cleaned Records** | **6,977** | **832** |
| **Time Range** | 2018-01 → 2026-01 | 2018-01-01 → 2020-01-29 |
| **Primary ML Targets** | `event_type`, `pilot_action`, `severity` | `problem_category`, `diverted_flag` |
| **Missing ML Targets** | 0 (100% complete) | 0 (100% complete) |

---

## 2. NASA ASRS ML Target Distributions

### 2.1 Operational Severity Rating (`severity`)

| Severity Tier | Count | Percentage | Operational Definition |
| :--- | :---: | :---: | :--- |
| **High** | 2,195 | 31.5% | Critical equipment failure, Loss of Control, NMAC, In-flight emergency diversion |
| **Medium** | 3,696 | 53.0% | Terrain proximity alert (CFTT/CFIT), Altitude excursion, Unstabilized approach, Weather encounter, Go-around |
| **Low** | 1,086 | 15.6% | Procedural deviation, Minor track/speed discrepancy, Informational ATC advisory |

### 2.2 Event Category (`event_type`)

| Event Category | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| Terrain / Obstacle Alert (CFIT/CFTT) | 1,920 | 27.52% | Hierarchical anomaly mapping |
| Track / Heading Deviation | 1,097 | 15.72% | Hierarchical anomaly mapping |
| Loss of Control | 926 | 13.27% | Hierarchical anomaly mapping |
| Altitude Deviation | 908 | 13.01% | Hierarchical anomaly mapping |
| Critical Equipment Problem | 667 | 9.56% | Hierarchical anomaly mapping |
| Near Mid-Air Collision (NMAC) | 518 | 7.42% | Hierarchical anomaly mapping |
| Weather / Turbulence Encounter | 368 | 5.27% | Hierarchical anomaly mapping |
| Speed Deviation | 336 | 4.82% | Hierarchical anomaly mapping |
| Unstabilized Approach | 237 | 3.40% | Hierarchical anomaly mapping |

### 2.3 Pilot Intervention Action (`pilot_action`)

| Pilot Action | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| Took Evasive Action | 1,155 | 16.55% | Crew response to anomaly |
| Returned to Clearance | 1,140 | 16.34% | Crew response to anomaly |
| Regained Aircraft Control | 846 | 12.13% | Crew response to anomaly |
| Overcame Equipment Problem | 664 | 9.52% | Crew response to anomaly |
| Became Reoriented | 646 | 9.26% | Crew response to anomaly |
| Executed Go-Around | 624 | 8.94% | Crew response to anomaly |
| ATC Intervention / Vectoring | 536 | 7.68% | Crew response to anomaly |
| No Action Reported | 443 | 6.35% | Crew response to anomaly |
| Requested ATC Assistance | 338 | 4.84% | Crew response to anomaly |
| Complied with Automation | 215 | 3.08% | Crew response to anomaly |
| Diverted Flight | 193 | 2.77% | Crew response to anomaly |
| Other Action | 177 | 2.54% | Crew response to anomaly |

---

## 3. Temporal Distribution (ASRS 2018–2026)

| Year | Incidents | % of Total | Cumulative |
| :---: | :---: | :---: | :---: |
| 2018 | 1,082 | 15.5% | 15.5% |
| 2019 | 1,009 | 14.5% | 30.0% |
| 2020 | 768 | 11.0% | 41.0% |
| 2021 | 845 | 12.1% | 53.1% |
| 2022 | 769 | 11.0% | 64.1% |
| 2023 | 779 | 11.2% | 75.3% |
| 2024 | 763 | 10.9% | 86.2% |
| 2025 | 882 | 12.6% | 98.9% |
| 2026 | 80 | 1.1% | 100.0% |

---

## 4. Operational Context

### 4.1 Flight Phase Distribution

| Flight Phase | Incidents | % of Reports |
| :--- | :---: | :---: |
| Initial Approach | 1,416 | 20.3% |
| Cruise | 1,160 | 16.6% |
| Descent | 1,096 | 15.7% |
| Final Approach | 1,002 | 14.4% |
| Climb | 861 | 12.3% |
| Initial Climb | 537 | 7.7% |
| Landing | 223 | 3.2% |
| Takeoff / Launch | 187 | 2.7% |
| Climb; Initial Climb | 54 | 0.8% |
| Final Approach; Initial Approach | 44 | 0.6% |

### 4.2 Top 10 Aircraft Types

| Make / Model | Incidents | % of Reports |
| :--- | :---: | :---: |
| Commercial Fixed Wing | 1,129 | 16.2% |
| B737 Undifferentiated or Other Model | 309 | 4.4% |
| EMB ERJ 170/175 ER/LR | 236 | 3.4% |
| B737-800 | 218 | 3.1% |
| B737-700 | 187 | 2.7% |
| EMB ERJ 145 ER/LR | 173 | 2.5% |
| Skyhawk 172/Cutlass 172 | 169 | 2.4% |
| Regional Jet 900 (CRJ900) | 152 | 2.2% |
| Light Transport; Low Wing; 2 Turbojet Eng | 149 | 2.1% |
| Medium Transport; Low Wing; 2 Turbojet Eng | 140 | 2.0% |

### 4.3 Primary Problem Assessment

| Primary Problem Factor | Incidents | % of Reports |
| :--- | :---: | :---: |
| Human Factors | 3,166 | 45.4% |
| Aircraft | 1,078 | 15.5% |
| Weather | 698 | 10.0% |
| Procedure | 682 | 9.8% |
| Ambiguous | 588 | 8.4% |
| Environment - Non Weather Related | 185 | 2.7% |
| Airspace Structure | 150 | 2.1% |
| Chart Or Publication | 146 | 2.1% |
| ATC Equipment / Nav Facility / Buildings | 97 | 1.4% |
| Software and Automation | 73 | 1.0% |

---

## 5. Cross-Tabulations

### 5.1 Severity vs Event Type

| event_type                           |   High |   Low |   Medium |   All |
|:-------------------------------------|-------:|------:|---------:|------:|
| Altitude Deviation                   |     10 |   170 |      728 |   908 |
| Critical Equipment Problem           |    667 |     0 |        0 |   667 |
| Loss of Control                      |    926 |     0 |        0 |   926 |
| Near Mid-Air Collision (NMAC)        |    518 |     0 |        0 |   518 |
| Speed Deviation                      |     12 |   215 |      109 |   336 |
| Terrain / Obstacle Alert (CFIT/CFTT) |     28 |     0 |     1892 |  1920 |
| Track / Heading Deviation            |      9 |   701 |      387 |  1097 |
| Unstabilized Approach                |      8 |     0 |      229 |   237 |
| Weather / Turbulence Encounter       |     17 |     0 |      351 |   368 |
| All                                  |   2195 |  1086 |     3696 |  6977 |

### 5.2 Pilot Action vs Event Type (Top 6 Actions)

| event_type                           |   Became Reoriented |   Executed Go-Around |   Overcame Equipment Problem |   Regained Aircraft Control |   Returned to Clearance |   Took Evasive Action |
|:-------------------------------------|--------------------:|---------------------:|-----------------------------:|----------------------------:|------------------------:|----------------------:|
| Altitude Deviation                   |                 122 |                   30 |                          106 |                          22 |                     279 |                   103 |
| Critical Equipment Problem           |                  11 |                   41 |                          172 |                          37 |                       6 |                    89 |
| Loss of Control                      |                   8 |                   80 |                           42 |                         679 |                      12 |                    41 |
| Near Mid-Air Collision (NMAC)        |                   6 |                   35 |                            4 |                           1 |                      10 |                   404 |
| Speed Deviation                      |                  43 |                   16 |                           42 |                           9 |                      45 |                    36 |
| Terrain / Obstacle Alert (CFIT/CFTT) |                 232 |                  229 |                           91 |                          43 |                     395 |                   358 |
| Track / Heading Deviation            |                 161 |                   57 |                          157 |                          11 |                     307 |                    77 |
| Unstabilized Approach                |                  30 |                  101 |                           16 |                          11 |                      18 |                     6 |
| Weather / Turbulence Encounter       |                  33 |                   35 |                           34 |                          33 |                      68 |                    41 |

---

## 6. OpenSky Metadata Profile

- **Total Emergencies Profiled:** 832
- **Distinct Aircraft Types (ICAO):** 97
- **Top Aircraft:** B738 (118), A320 (112), A319 (52), A321 (40), B763 (35)
- **Diverted Rate:** 295 flights (35.5%)
- **Fuel Dumping Rate:** 32 flights (3.8%)

### Top Emergency Problem Categories (Crowdsourced + Aviation Herald)

| Category | Flights | % of Total |
| :--- | :---: | :---: |
| unspecified | 412 | 49.5% |
| unconfirmed_emergency | 209 | 25.1% |
| medical_emergency | 74 | 8.9% |
| engine_problem | 25 | 3.0% |
| depressurization_cabin | 25 | 3.0% |
| technical_systems | 25 | 3.0% |
| smoke_fire_fumes | 21 | 2.5% |
| gear_hydraulics | 12 | 1.4% |
| fuel_issue | 8 | 1.0% |
| cracked_windshield | 5 | 0.6% |
