# Student Performance Analytics System
## Project Documentation & Academic Report

**Course**: EWB Courses – Python with AI  
**Assignment**: Python with AI Practical Project  
**Author**: Student / Developer  
**Status**: Completed, Tested & Validated  
**GitHub Repository**: [`https://github.com/srikarbabuvaddi-star/EWB-PythonAI`](https://github.com/srikarbabuvaddi-star/EWB-PythonAI)  

---

## 1. Executive Summary & Project Objective

The **Student Performance Analytics System** is a software solution engineered in Python using **NumPy** and **Pandas** to analyze, evaluate, and report student academic performance and institutional attendance compliance.

### Core Objectives:
1. **Automated Data Processing**: Ingest a structured CSV dataset of student examination scores and attendance records.
2. **Data Integrity Assurance**: Perform validation checks to catch missing values, duplicate identifiers, out-of-range marks, and corrupted data types.
3. **High-Performance Numerical Computation**: Leverage NumPy vectorized operations for row-wise sums, means, and subject-level aggregations.
4. **Transparent Academic Grading**: Implement a clearly defined pass/fail rule (minimum 35 marks in each subject) alongside a documented letter-grade scale ($A$ through $F$).
5. **Fair Rank Attribution**: Resolve score ties accurately using dense ranking so that top-performing students are not arbitrarily truncated.
6. **Attendance Auditing**: Evaluate attendance as a separate compliance parameter, isolating administrative warnings from academic marks.
7. **Automated Evidence-Based Insights**: Synthesize key analytical conclusions directly from computed results without hardcoded assertions.

---

## 2. Dataset Description

The dataset `students.csv` contains **22 fictional student records** representing undergraduate students across four academic departments: Computer Science, Information Technology, Data Science, and Artificial Intelligence.

### Data Schema:

| Column Name | Data Type | Value Range | Description |
| :--- | :--- | :--- | :--- |
| `Student_ID` | String / Object | Unique `STD101`–`STD122` | Primary unique student identifier |
| `Name` | String / Object | Full Name | Fictional student name |
| `Department` | String / Object | Categorical | Academic department |
| `Python` | Numeric (Float) | $0.0$ to $100.0$ | Marks scored in Python Programming (out of 100) |
| `Data_Structures` | Numeric (Float) | $0.0$ to $100.0$ | Marks scored in Data Structures (out of 100) |
| `DBMS` | Numeric (Float) | $0.0$ to $100.0$ | Marks scored in Database Management Systems (out of 100) |
| `Statistics` | Numeric (Float) | $0.0$ to $100.0$ | Marks scored in Applied Statistics (out of 100) |
| `AI_Basics` | Numeric (Float) | $0.0$ to $100.0$ | Marks scored in Artificial Intelligence Fundamentals (out of 100) |
| `Attendance` | Numeric (Float) | $0.0$ to $100.0$ | Overall attendance percentage |

### Dataset Design Considerations:
- **Grade Distribution**: Calibrated to cover all letter grades ($A$, $B$, $C$, $D$, $E$, and $F$).
- **Pass/Fail Edge Cases**: Includes students with high overall averages who failed exactly one subject (e.g., STD115 scored $85, 88, 82, 84$ in four subjects, but $28$ in Statistics; overall average $73.4\%$, yet graded $F$).
- **Tie Conditions**: Students STD103 (Rohan Iyer) and STD104 (Ananya Nair) both share identical marks in all 5 subjects and equal average scores ($88.8\%$) to validate tie handling.
- **Attendance Variation**: 5 students exhibit attendance below $75.0\%$ to evaluate the attendance warning trigger.

---

## 3. Technologies & Software Architecture

### Technology Stack:
- **Python 3.12**: Core programming language using clean functional paradigms.
- **NumPy**: Fast vectorized array calculations (`np.sum`, `np.mean`, `np.min`, `np.max`, `np.std`).
- **Pandas**: Tabular DataFrame operations, filtering, grouping, sorting, and CSV parsing.
- **Tabulate**: Presentation-grade ASCII table rendering.
- **PyTest**: Automated test execution framework for unit and integration testing.

### Architectural Diagram:

```
┌────────────────────────────────────────────────────────┐
│                   students.csv                         │
│            (22 Fictional Student Records)              │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               load_student_dataset()                   │
│        (Pandas Ingestion & File Verification)          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                validate_student_data()                 │
│  - Required Columns Check  - Missing / Null Values     │
│  - Duplicate ID Detection  - Marks & Attendance Range  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             calculate_student_metrics()                │
│       (NumPy Vectorized Sum & Mean: Axis=1)            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             assign_grades_and_status()                 │
│  - Subject Threshold (>= 35) -> Pass/Fail              │
│  - Grade Mapping (A, B, C, D, E, F)                    │
└──────────────────────────┬─────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Class Stats  │  │ Subject-Wise  │  │   Top Ranks   │
│ (High, Low,   │  │   Analysis    │  │  (Dense Rank  │
│  Mean, Std)   │  │ (Mean, Rates) │  │  With Ties)   │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│       Attendance Auditing & Key Observations           │
│      - Attendance < 75% Alerts                         │
│      - Data-Driven Synthesis & Terminal Report         │
└────────────────────────────────────────────────────────┘
```

---

## 4. Function Breakdown & Implementation Details

All core computational and analytical logic is encapsulated in modular functions within `functions.py`:

### 4.1. `load_student_dataset(filepath: str) -> pd.DataFrame`
- **Purpose**: Reads CSV data into a Pandas DataFrame.
- **Error Handling**: Raises `FileNotFoundError` if the path does not exist and `ValueError` if the file is blank or corrupted.

### 4.2. `validate_student_data(df, required_columns, subject_columns, raise_on_error) -> dict`
- **Purpose**: Verifies that the dataset meets all five validation criteria:
  1. Checks for all mandatory column headers.
  2. Detects `null` / `NaN` cells across any record.
  3. Ensures `Student_ID` values are strictly unique.
  4. Checks that subject marks are numeric and within $[0.0, 100.0]$.
  5. Checks that attendance percentages are numeric and within $[0.0, 100.0]$.
- **Return**: A dictionary containing `is_valid`, `errors`, and `total_records`. Raises `ValueError` if errors exist and `raise_on_error=True`.

### 4.3. `calculate_student_metrics(df, subject_columns) -> pd.DataFrame`
- **NumPy Usage**: Extracts subject marks as a 2D float NumPy matrix:
  ```python
  marks_matrix = df[subject_columns].to_numpy(dtype=float)
  total_marks = np.sum(marks_matrix, axis=1)
  average_marks = np.mean(marks_matrix, axis=1)
  ```
- **Benefit**: Replaces slow Python row iterations with SIMD-optimized C-level array math.

### 4.4. `determine_single_student_grade(average_mark, subject_marks, pass_threshold) -> tuple`
- **Purpose**: Evaluates academic rules for an individual student:
  - If `any(mark < 35.0 for mark in subject_marks)`: Returns `("Fail", "F")`.
  - Else assigns letter grade by average:
    - $\ge 90\% \rightarrow A$
    - $\ge 75\% \rightarrow B$
    - $\ge 60\% \rightarrow C$
    - $\ge 50\% \rightarrow D$
    - $\ge 35\% \rightarrow E$
    - $< 35\% \rightarrow F$

### 4.5. `assign_grades_and_status(df, subject_columns, pass_threshold) -> pd.DataFrame`
- **Purpose**: Applies `determine_single_student_grade` across the DataFrame, appending `Status` and `Grade` columns.

### 4.6. `calculate_class_statistics(df, subject_columns) -> dict`
- **Purpose**: Computes cohort-wide summary statistics using NumPy:
  - `np.max`, `np.min`, `np.mean`, and `np.std`.
  - Computes pass count, fail count, pass percentage, and full grade distribution.

### 4.7. `calculate_subject_analysis(df, subject_columns, pass_threshold) -> pd.DataFrame`
- **Purpose**: Generates subject-level breakdowns:
  - Computes Mean, Highest, Lowest, Standard Deviation, and Pass Rate for each subject.

### 4.8. `get_top_performers(df, top_n=3, rank_by='Average_Marks') -> pd.DataFrame`
- **Purpose**: Identifies top-performing students and handles ties.
- **Tie Handling Implementation**:
  ```python
  df["Rank"] = df[rank_by].rank(method="dense", ascending=False).astype(int)
  top_df = df[df["Rank"] <= top_n]
  ```
  *Dense ranking* ensures that if two students tie for Rank 1, both are assigned Rank 1, and the next score tier receives Rank 2. No tied student is arbitrarily omitted.

### 4.9. `analyze_attendance(df, threshold=75.0) -> dict`
- **Purpose**: Isolates attendance monitoring.
- **Calculations**: Mean attendance, max/min, count of students below $75\%$, and extracts a DataFrame of students needing attendance counseling.

### 4.10. `generate_key_observations(...) -> list`
- **Purpose**: Dynamically constructs analytical takeaways based strictly on the computed DataFrame and statistical dictionaries.

---

## 5. NumPy vs. Pandas: Practical Comparison & Usage

This project purposefully demonstrates when and why to use **NumPy** versus **Pandas**:

| Feature / Task | Used Library | Rationale |
| :--- | :--- | :--- |
| **Data Ingestion & CSV Parsing** | **Pandas** | High-level `pd.read_csv` handles headers, types, and tabular representations effortlessly. |
| **Row-wise Totals & Averages** | **NumPy** | Converting subject columns into a 2D matrix (`.to_numpy()`) and calculating `np.sum(axis=1)` / `np.mean(axis=1)` is computationally optimal. |
| **Data Filtering & Subsetting** | **Pandas** | Expressive query syntax like `df[df["Attendance"] < 75.0]` provides clean, readable code. |
| **Subject Aggregations** | **NumPy** | `np.mean()`, `np.std()`, `np.min()`, `np.max()` provide fast mathematical primitives on contiguous arrays. |
| **Ranking with Ties** | **Pandas** | Built-in `.rank(method='dense')` provides clean, robust competition ranking. |

---

## 6. Project Grading Rules & Policy Rationale

> [!IMPORTANT]
> **Academic Model Notice**: These grading rules are project-defined for the EWB assignment and do not represent official university or board regulations.

### Defined Rules:
1. **Subjects**: 5 subjects, each scored out of 100. Total maximum possible marks: 500.
2. **Subject Minimum Cutoff**: Every subject requires at least **35 marks** to clear the subject.
3. **Pass / Fail Condition**:
   - $\text{Pass}$: Student scores $\ge 35$ in **all 5 subjects**.
   - $\text{Fail}$: Student scores $< 35$ in **one or more subjects**.
4. **Letter Grades**:
   - **Grade A**: Average $\ge 90.0\%$
   - **Grade B**: Average $\ge 75.0\%$ and $< 90.0\%$
   - **Grade C**: Average $\ge 60.0\%$ and $< 75.0\%$
   - **Grade D**: Average $\ge 50.0\%$ and $< 60.0\%$
   - **Grade E**: Average $\ge 35.0\%$ and $< 50.0\%$
   - **Grade F**: Scored $< 35$ in any individual subject OR overall average $< 35.0\%$.
5. **Decoupled Attendance**:
   - Attendance is audited against a $75.0\%$ institutional standard.
   - It triggers attendance warnings but does not alter the academic grade.

---

## 7. Actual Execution Results & Output Evidence

Below is the verbatim terminal output produced by executing `python main.py` in the workspace environment:

```text
********************************************************************************
********************* STUDENT PERFORMANCE ANALYTICS SYSTEM *********************
********************************************************************************
Project : EWB Courses - Python with AI Assignment
Tech    : Python 3 | NumPy | Pandas | Tabulate
Rules   : Project-Defined Grading Rules (Academic Model)
--------------------------------------------------------------------------------

[Step 1/7] Loading dataset from CSV file...
-> Successfully loaded 22 student records from 'C:\Users\hi\Desktop\EWB PythonAI\students.csv'.

[Step 2/7] Validating data integrity...
-> Validation passed: 0 missing columns, 0 null cells, 0 duplicate IDs, 0 out-of-range marks.

================================================================================
=============== 1. DATASET OVERVIEW (FIRST 5 FICTIONAL RECORDS) ================
================================================================================
+--------------+-----------------+-------------------------+----------+-------------------+--------+--------------+-------------+--------------+
| Student_ID   | Name            | Department              |   Python |   Data_Structures |   DBMS |   Statistics |   AI_Basics |   Attendance |
+==============+=================+=========================+==========+===================+========+==============+=============+==============+
| STD101       | Aarav Sharma    | Computer Science        |       95 |                92 |     94 |           96 |          98 |         96   |
+--------------+-----------------+-------------------------+----------+-------------------+--------+--------------+-------------+--------------+
| STD102       | Diya Patel      | Artificial Intelligence |       92 |                94 |     90 |           98 |          91 |         94.5 |
+--------------+-----------------+-------------------------+----------+-------------------+--------+--------------+-------------+--------------+
| STD103       | Rohan Iyer      | Data Science            |       88 |                85 |     90 |           92 |          89 |         91   |
+--------------+-----------------+-------------------------+----------+-------------------+--------+--------------+-------------+--------------+
| STD104       | Ananya Nair     | Information Technology  |       88 |                85 |     90 |           92 |          89 |         89   |
+--------------+-----------------+-------------------------+----------+-------------------+--------+--------------+-------------+--------------+
| STD105       | Vikram Malhotra | Computer Science        |       82 |                78 |     85 |           80 |          84 |         86.5 |
+--------------+-----------------+-------------------------+----------+-------------------+--------+--------------+-------------+--------------+

Total Records: 22 | Total Attributes: 9
Subjects Evaluated (out of 100): Python, Data_Structures, DBMS, Statistics, AI_Basics

[Step 3/7] Calculating totals, averages, and applying grading rules...

================================================================================
============ 2. STUDENT RESULTS (TOTALS, AVERAGES, GRADES & STATUS) ============
================================================================================
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| Student_ID   | Name            | Department              |   Total_Marks |   Average_Marks | Grade   | Status   |   Attendance |
+==============+=================+=========================+===============+=================+=========+==========+==============+
| STD101       | Aarav Sharma    | Computer Science        |           475 |            95   | A       | Pass     |         96   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD102       | Diya Patel      | Artificial Intelligence |           465 |            93   | A       | Pass     |         94.5 |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD103       | Rohan Iyer      | Data Science            |           444 |            88.8 | B       | Pass     |         91   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD104       | Ananya Nair     | Information Technology  |           444 |            88.8 | B       | Pass     |         89   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD105       | Vikram Malhotra | Computer Science        |           409 |            81.8 | B       | Pass     |         86.5 |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD106       | Sneha Kulkarni  | Data Science            |           383 |            76.6 | B       | Pass     |         73   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD107       | Arjun Verma     | Artificial Intelligence |           353 |            70.6 | C       | Pass     |         85   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD108       | Pooja Hegde     | Information Technology  |           334 |            66.8 | C       | Pass     |         82   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD109       | Rahul Sen       | Computer Science        |           315 |            63   | C       | Pass     |         70.5 |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD110       | Meera Krishnan  | Data Science            |           283 |            56.6 | D       | Pass     |         88   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD111       | Kabir Das       | Artificial Intelligence |           266 |            53.2 | D       | Pass     |         78   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD112       | Tanvi Deshmukh  | Information Technology  |           236 |            47.2 | E       | Pass     |         80   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD113       | Aditya Joshi    | Computer Science        |           210 |            42   | E       | Pass     |         67   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD114       | Ritu Banerjee   | Data Science            |           192 |            38.4 | E       | Pass     |         76   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD115       | Nikhil Gupta    | Artificial Intelligence |           367 |            73.4 | F       | Fail     |         90   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD116       | Ishita Roy      | Information Technology  |           305 |            61   | F       | Fail     |         74   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD117       | Siddharth Rao   | Computer Science        |           180 |            36   | F       | Fail     |         62   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD118       | Kavya Menon     | Data Science            |           460 |            92   | A       | Pass     |         95   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD119       | Varun Reddy     | Artificial Intelligence |           386 |            77.2 | B       | Pass     |         84   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD120       | Neha Choudhury  | Information Technology  |           305 |            61   | C       | Pass     |         87   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD121       | Devendra Singh  | Computer Science        |           261 |            52.2 | D       | Pass     |         79   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+
| STD122       | Priya Nambiar   | Data Science            |           430 |            86   | B       | Pass     |         92   |
+--------------+-----------------+-------------------------+---------------+-----------------+---------+----------+--------------+

PROJECT-DEFINED GRADING RULES EXPLANATION:
  * Note: These are project-defined assignment rules, not official university rules.
  * Pass Condition: Minimum 35 marks required in EVERY subject.
  * Fail Rule     : Any subject < 35 marks results in Grade 'F' and Status 'Fail'.
  * Grade Scale   : A (>=90%), B (75-89%), C (60-74%), D (50-59%), E (35-49%), F (<35% or subject fail).
  * Attendance    : Tracked separately as institutional compliance (does not alter academic grade).

[Step 4/7] Generating class-wide performance metrics...

================================================================================
======================= 3. CLASS PERFORMANCE STATISTICS ========================
================================================================================
+-------------------------------+--------------------------------+
| Metric                        | Value                          |
+===============================+================================+
| Total Enrolled Students       | 22                             |
+-------------------------------+--------------------------------+
| Students Passed               | 19 (86.36%)                    |
+-------------------------------+--------------------------------+
| Students Failed               | 3 (13.64%)                     |
+-------------------------------+--------------------------------+
| Highest-Performing Student(s) | Aarav Sharma (STD101) [95.0%]  |
+-------------------------------+--------------------------------+
| Lowest-Performing Student(s)  | Siddharth Rao (STD117) [36.0%] |
+-------------------------------+--------------------------------+
| Highest Total Marks           | 475.0 / 500.0                  |
+-------------------------------+--------------------------------+
| Lowest Total Marks            | 180.0 / 500.0                  |
+-------------------------------+--------------------------------+
| Highest Average Mark          | 95.0%                          |
+-------------------------------+--------------------------------+
| Lowest Average Mark           | 36.0%                          |
+-------------------------------+--------------------------------+
| Class Overall Mean Average    | 68.21%                         |
+-------------------------------+--------------------------------+
| Class Standard Deviation      | 18.13%                         |
+-------------------------------+--------------------------------+

Grade Distribution:
Grade      Student Count  Percentage
-------  ---------------  ------------
A                      3  13.6%
B                      6  27.3%
C                      4  18.2%
D                      3  13.6%
E                      3  13.6%
F                      3  13.6%

[Step 5/7] Analyzing subject-wise performance...

================================================================================
===================== 4. SUBJECT-WISE PERFORMANCE ANALYSIS =====================
================================================================================
+-----------------+-----------------+----------------+---------------+-----------------+---------------+
| Subject         |   Average_Marks |   Highest_Mark |   Lowest_Mark |   Std_Deviation |   Pass_Rate_% |
+=================+=================+================+===============+=================+===============+
| Python          |           66.41 |             95 |            25 |           20.9  |         90.91 |
+-----------------+-----------------+----------------+---------------+-----------------+---------------+
| Data_Structures |           67.18 |             94 |            32 |           18.36 |         95.45 |
+-----------------+-----------------+----------------+---------------+-----------------+---------------+
| DBMS            |           70.45 |             94 |            42 |           17.21 |        100    |
+-----------------+-----------------+----------------+---------------+-----------------+---------------+
| Statistics      |           67.64 |             98 |            28 |           21.11 |         95.45 |
+-----------------+-----------------+----------------+---------------+-----------------+---------------+
| AI_Basics       |           69.36 |             98 |            39 |           17.95 |        100    |
+-----------------+-----------------+----------------+---------------+-----------------+---------------+

* Best-Performing Subject : 'DBMS' (Class Mean: 70.45%, Pass Rate: 100.0%)
* Most Challenging Subject: 'Python' (Class Mean: 66.41%, Pass Rate: 90.91%)

[Step 6/7] Identifying top performers with tie-aware ranking...

================================================================================
============== 5. TOP PERFORMING STUDENTS (TOP 3 RANKS WITH TIES) ==============
================================================================================
+--------+--------------+--------------+-------------------------+---------------+-----------------+---------+----------+
|   Rank | Student_ID   | Name         | Department              |   Total_Marks |   Average_Marks | Grade   | Status   |
+========+==============+==============+=========================+===============+=================+=========+==========+
|      1 | STD101       | Aarav Sharma | Computer Science        |           475 |              95 | A       | Pass     |
+--------+--------------+--------------+-------------------------+---------------+-----------------+---------+----------+
|      2 | STD102       | Diya Patel   | Artificial Intelligence |           465 |              93 | A       | Pass     |
+--------+--------------+--------------+-------------------------+---------------+-----------------+---------+----------+
|      3 | STD118       | Kavya Menon  | Data Science            |           460 |              92 | A       | Pass     |
+--------+--------------+--------------+-------------------------+---------------+-----------------+---------+----------+

Tie-Handling Note:
  Uses dense ranking: Students with identical average scores share the exact same rank.
  All students in the top 3 score tiers are included without unfair truncation.

[Step 7/7] Analyzing attendance records and generating key insights...

================================================================================
================= 6. ATTENDANCE SUMMARY & LOW ATTENDANCE ALERT =================
================================================================================
+----------------------------------+-------------+
| Attendance Indicator             | Value       |
+==================================+=============+
| Class Average Attendance         | 82.7%       |
+----------------------------------+-------------+
| Highest Attendance               | 96.0%       |
+----------------------------------+-------------+
| Lowest Attendance                | 62.0%       |
+----------------------------------+-------------+
| Compliance Threshold             | 75.0%       |
+----------------------------------+-------------+
| Compliant Students (>= 75%)      | 17 (77.27%) |
+----------------------------------+-------------+
| Students Below Threshold (< 75%) | 5 (22.73%)  |
+----------------------------------+-------------+

[!] WARNING: The following 5 student(s) have fallen below the 75% attendance threshold:
+--------------+----------------+------------------------+--------------+-----------------+----------+
| Student_ID   | Name           | Department             |   Attendance |   Average_Marks | Status   |
+==============+================+========================+==============+=================+==========+
| STD117       | Siddharth Rao  | Computer Science       |         62   |            36   | Fail     |
+--------------+----------------+------------------------+--------------+-----------------+----------+
| STD113       | Aditya Joshi   | Computer Science       |         67   |            42   | Pass     |
+--------------+----------------+------------------------+--------------+-----------------+----------+
| STD109       | Rahul Sen      | Computer Science       |         70.5 |            63   | Pass     |
+--------------+----------------+------------------------+--------------+-----------------+----------+
| STD106       | Sneha Kulkarni | Data Science           |         73   |            76.6 | Pass     |
+--------------+----------------+------------------------+--------------+-----------------+----------+
| STD116       | Ishita Roy     | Information Technology |         74   |            61   | Fail     |
+--------------+----------------+------------------------+--------------+-----------------+----------+

================================================================================
============ 7. KEY ANALYTICAL OBSERVATIONS (DATA-DRIVEN INSIGHTS) =============
================================================================================
  1. Overall Class Performance: 19 out of 22 students (86.36%) cleared all academic requirements.
  2. Strongest Subject: 'DBMS' recorded the highest average of 70.45 marks with a 100.0% pass rate.
  3. Most Challenging Subject: 'Python' recorded the lowest average of 66.41 marks with a 90.91% pass rate.
  4. Impact of Subject Minimum Rule (35 marks): Nikhil Gupta (Avg: 73.4%), Ishita Roy (Avg: 61.0%) achieved solid overall averages but received Grade F due to falling below 35 in a single subject, demonstrating strict policy compliance.
  5. Attendance vs Academics Correlation: Students with >= 75% attendance averaged 71.88 marks, compared to 55.72 marks for students with low attendance (< 75%), showing a 16.16 mark performance advantage for regular attendees.
  6. Institutional Compliance: 5 student(s) (22.73%) breached the 75% attendance threshold and require institutional counseling.

================================================================================
###### Analytics processing complete. All outputs generated successfully! ######
================================================================================
```

---

## 8. Automated Testing & Verification Evidence

The test suite in `test_analytics.py` executes **38 automated tests** spanning 8 testing categories:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\hi\Desktop\EWB PythonAI
collected 38 items

test_analytics.py::test_calculate_student_metrics PASSED                 [  2%]
test_analytics.py::test_grade_boundaries[90.0-marks0-Pass-A] PASSED      [  5%]
test_analytics.py::test_grade_boundaries[95.5-marks1-Pass-A] PASSED      [  7%]
test_analytics.py::test_grade_boundaries[89.99-marks2-Pass-B] PASSED     [ 10%]
test_analytics.py::test_grade_boundaries[75.0-marks3-Pass-B] PASSED      [ 13%]
test_analytics.py::test_grade_boundaries[85.0-marks4-Pass-B] PASSED      [ 15%]
test_analytics.py::test_grade_boundaries[74.99-marks5-Pass-C] PASSED     [ 18%]
test_analytics.py::test_grade_boundaries[60.0-marks6-Pass-C] PASSED      [ 21%]
test_analytics.py::test_grade_boundaries[68.5-marks7-Pass-C] PASSED      [ 23%]
test_analytics.py::test_grade_boundaries[59.99-marks8-Pass-D] PASSED     [ 26%]
test_analytics.py::test_grade_boundaries[50.0-marks9-Pass-D] PASSED      [ 28%]
test_analytics.py::test_grade_boundaries[55.0-marks10-Pass-D] PASSED     [ 31%]
test_analytics.py::test_grade_boundaries[49.99-marks11-Pass-E] PASSED    [ 34%]
test_analytics.py::test_grade_boundaries[35.0-marks12-Pass-E] PASSED     [ 36%]
test_analytics.py::test_grade_boundaries[45.0-marks13-Pass-E] PASSED     [ 39%]
test_analytics.py::test_grade_boundaries[34.99-marks14-Fail-F] PASSED    [ 42%]
test_analytics.py::test_pass_condition_all_above_threshold PASSED        [ 44%]
test_analytics.py::test_fail_condition_single_subject_below_threshold PASSED [ 47%]
test_analytics.py::test_assign_grades_and_status_dataframe PASSED        [ 50%]
test_analytics.py::test_calculate_subject_analysis PASSED                [ 52%]
test_analytics.py::test_top_performers_and_ties PASSED                   [ 55%]
test_analytics.py::test_validation_missing_required_column PASSED        [ 57%]
test_analytics.py::test_validation_missing_null_values PASSED            [ 60%]
test_analytics.py::test_validation_duplicate_student_ids PASSED          [ 63%]
test_analytics.py::test_validation_invalid_marks_negative PASSED         [ 65%]
test_analytics.py::test_validation_invalid_marks_above_100 PASSED        [ 68%]
test_analytics.py::test_validation_invalid_marks_non_numeric PASSED      [ 71%]
test_analytics.py::test_attendance_analysis PASSED                       [ 73%]
test_analytics.py::test_edge_case_empty_dataset_validation PASSED        [ 76%]
test_analytics.py::test_edge_case_empty_dataset_class_statistics PASSED  [ 78%]
test_analytics.py::test_edge_case_missing_student_names PASSED           [ 81%]
test_analytics.py::test_edge_case_missing_student_ids PASSED             [ 84%]
test_analytics.py::test_edge_case_invalid_attendance_negative PASSED     [ 86%]
test_analytics.py::test_edge_case_invalid_attendance_above_100 PASSED    [ 89%]
test_analytics.py::test_edge_case_invalid_attendance_non_numeric PASSED  [ 92%]
test_analytics.py::test_edge_case_no_passing_students PASSED             [ 94%]
test_analytics.py::test_edge_case_missing_csv_file PASSED                [ 97%]
test_analytics.py::test_edge_case_unreadable_or_empty_csv_file PASSED    [100%]

============================= 38 passed in 2.27s =============================
```

---

## 9. Data Integrity & Ethical Standards

1. **Zero Fabrication**: All analytics, averages, rates, and observations are dynamically calculated using verified NumPy and Pandas operations. No figures are hardcoded.
2. **Explicit Practice Data**: All 22 records in `students.csv` are strictly fictional practice records generated for educational benchmarking.
3. **No Silent Replacement of Corrupted Data**: The system strictly raises transparent, descriptive validation exceptions rather than silently imputing or omitting invalid records, ensuring errors are fixed at the source.
4. **Privacy Preservation**: No real personal identifiers, student contact information, or sensitive university records are stored or exposed.

---

## 10. Challenges Encountered & Key Learnings

1. **Windows Console Character Map (`cp1252`)**:
   - *Challenge*: Default Windows command consoles choke on Unicode box borders used by some terminal formatters.
   - *Resolution*: Implemented `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` and ASCII-safe `grid` formatting, guaranteeing 100% crash-free execution.
2. **Dense-Rank Tie Handling**:
   - *Challenge*: Simple `.head(3)` drops students who achieve tied scores at rank cutoffs.
   - *Resolution*: Implemented `rank(method='dense')` to ensure equal recognition for tied scores.
3. **Strict Subject Cutoff Policy**:
   - *Challenge*: Ensuring a high average does not bypass failing an individual subject.
   - *Resolution*: Decoupled subject pass evaluation from aggregate average calculation.
4. **Edge-Case Resilience**:
   - *Challenge*: Preventing division-by-zero on empty datasets or zero-passer cohorts.
   - *Resolution*: Added defensive input validation and mathematical guards across all analytical modules.

---

## 11. Trainer Presentation Guide & Demonstration Script

When presenting this project to your instructor or trainer:

1. **Introduction**:
   > *"I have built a Student Performance Analytics System using Python, NumPy, and Pandas, structured into modular files (`functions.py`, `main.py`, `students.csv`, and `test_analytics.py`)."*
2. **Demonstrate Data Integrity**:
   > *"Before running analytics, our system validates the CSV for missing values, duplicate IDs, non-numeric marks, blank names, and out-of-range scores to protect downstream calculations."*
3. **Highlight NumPy Utilization**:
   > *"Rather than relying on slow nested Python loops for numerical math, we convert our subject columns into a 2D NumPy array to perform vectorized row sums and means, as well as subject-level standard deviations and aggregations."*
4. **Explain Academic Grading Rules**:
   > *"The assignment defines that a student must achieve at least 35 in every single subject. We can observe student STD115 (Nikhil Gupta): despite having a 73.4% average across all subjects, scoring 28 in Statistics triggered a Grade F. This proves our conditional logic works accurately."*
5. **Demonstrate Tie Handling**:
   > *"We implemented dense ranking for top performers so that students with identical scores share the same rank and are not unfairly truncated."*
6. **Show Test Results**:
   > *"We have an automated PyTest suite with 38 tests verifying boundary cases, invalid inputs, edge cases (empty data, missing files, zero passers), and calculation accuracy. All 38 tests pass in under 3 seconds."*
