"""
Student Performance Analytics System - Core Functions Module
=============================================================
This module provides reusable, modular functions for loading, validating,
analyzing, and reporting student academic performance and attendance data.

Built for the EWB Courses Python with AI project assignment.
Technologies: Python 3, NumPy, Pandas
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate

# Global default configuration constants
DEFAULT_SUBJECTS = ["Python", "Data_Structures", "DBMS", "Statistics", "AI_Basics"]
DEFAULT_REQUIRED_COLUMNS = ["Student_ID", "Name", "Department", "Attendance"] + DEFAULT_SUBJECTS
PASS_MARK_THRESHOLD = 35.0
MAX_MARK_LIMIT = 100.0
MIN_MARK_LIMIT = 0.0
ATTENDANCE_WARNING_THRESHOLD = 75.0


# ============================================================================
# 1. DATA LOADING FUNCTION
# ============================================================================

def load_student_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads student records from a CSV file into a Pandas DataFrame.

    Parameters:
        filepath (str): Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: Loaded student records.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Dataset file not found at '{filepath}'")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file '{filepath}': {str(e)}")

    if df.empty:
        raise ValueError(f"Error: Dataset file '{filepath}' is empty.")

    return df


# ============================================================================
# 2. DATA VALIDATION FUNCTION
# ============================================================================

def validate_student_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    subject_columns: Optional[List[str]] = None,
    raise_on_error: bool = True
) -> Dict[str, Any]:
    """
    Validates student dataset integrity based on required checks:
      1. Presence of required columns.
      2. Missing / null values in any column.
      3. Duplicate Student IDs.
      4. Invalid marks (< 0 or > 100 or non-numeric).
      5. Invalid attendance (< 0 or > 100 or non-numeric).

    Parameters:
        df (pd.DataFrame): Input dataframe to validate.
        required_columns (list): List of required column headers.
        subject_columns (list): List of subject column headers.
        raise_on_error (bool): If True, raises ValueError upon failure.

    Returns:
        dict: Summary of validation findings (is_valid, errors, warnings, row_count).

    Raises:
        ValueError: If raise_on_error is True and any validation error is detected.
    """
    if required_columns is None:
        required_columns = DEFAULT_REQUIRED_COLUMNS
    if subject_columns is None:
        subject_columns = DEFAULT_SUBJECTS

    errors = []
    warnings = []

    # 0. Empty DataFrame check
    if df.empty or len(df) == 0:
        errors.append("Dataset is empty. At least one student record is required.")
        if raise_on_error:
            raise ValueError("; ".join(errors))
        return {
            "is_valid": False,
            "total_records": 0,
            "errors": errors,
            "warnings": warnings
        }

    # 1. Required columns check
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in dataset: {missing_cols}")

    if errors and raise_on_error:
        raise ValueError("; ".join(errors))

    # 2. Missing (null/NaN) values check
    null_counts = df[required_columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_details = cols_with_nulls.to_dict()
        errors.append(f"Found missing/null values in columns: {null_details}")

    # 3. Blank or whitespace-only Student IDs and Names check
    if "Student_ID" in df.columns:
        blank_ids = df["Student_ID"].astype(str).str.strip().isin(["", "nan", "None"])
        if blank_ids.any():
            errors.append("Student dataset contains blank or missing Student IDs.")

    if "Name" in df.columns:
        blank_names = df["Name"].astype(str).str.strip().isin(["", "nan", "None"])
        if blank_names.any():
            errors.append("Student dataset contains blank or missing Student Names.")

    # 4. Duplicate Student IDs check
    if "Student_ID" in df.columns:
        clean_ids = df["Student_ID"].astype(str).str.strip()
        duplicate_ids = clean_ids[clean_ids.duplicated(keep=False)].tolist()
        if duplicate_ids:
            unique_dupes = [d for d in set(duplicate_ids) if d not in ["", "nan", "None"]]
            if unique_dupes:
                errors.append(f"Duplicate Student IDs detected: {unique_dupes}")

    # 5. Subject marks range check [0, 100]
    for sub in subject_columns:
        if sub in df.columns:
            # Check for non-numeric data
            non_numeric = pd.to_numeric(df[sub], errors="coerce").isnull() & df[sub].notnull()
            if non_numeric.any():
                errors.append(f"Subject '{sub}' contains non-numeric mark entries.")
                continue

            marks = df[sub].astype(float)
            out_of_bounds = df[(marks < MIN_MARK_LIMIT) | (marks > MAX_MARK_LIMIT)]
            if not out_of_bounds.empty:
                bad_records = out_of_bounds[["Student_ID", sub]].to_dict(orient="records")
                errors.append(f"Subject '{sub}' has marks outside [0, 100]: {bad_records}")

    # 6. Attendance range check [0, 100]
    if "Attendance" in df.columns:
        att_non_numeric = pd.to_numeric(df["Attendance"], errors="coerce").isnull() & df["Attendance"].notnull()
        if att_non_numeric.any():
            errors.append("Attendance column contains non-numeric entries.")
        else:
            att = df["Attendance"].astype(float)
            out_of_bounds_att = df[(att < 0.0) | (att > 100.0)]
            if not out_of_bounds_att.empty:
                bad_att = out_of_bounds_att[["Student_ID", "Attendance"]].to_dict(orient="records")
                errors.append(f"Attendance percentages outside [0, 100]: {bad_att}")

    is_valid = len(errors) == 0

    validation_result = {
        "is_valid": is_valid,
        "total_records": len(df),
        "errors": errors,
        "warnings": warnings
    }

    if not is_valid and raise_on_error:
        raise ValueError("Data Validation Failed:\n- " + "\n- ".join(errors))

    return validation_result


# ============================================================================
# 3. METRIC CALCULATIONS USING NUMPY
# ============================================================================

def calculate_student_metrics(
    df: pd.DataFrame,
    subject_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates Total Marks and Average Marks for each student using NumPy arrays.

    Parameters:
        df (pd.DataFrame): Validated student dataframe.
        subject_columns (list): List of subject column names.

    Returns:
        pd.DataFrame: New DataFrame with added 'Total_Marks' and 'Average_Marks' columns.
    """
    if subject_columns is None:
        subject_columns = DEFAULT_SUBJECTS

    df_result = df.copy()

    # Extract subject marks as a 2D NumPy array (shape: n_students x 5)
    marks_matrix = df_result[subject_columns].to_numpy(dtype=float)

    # Calculate Total Marks across axis 1 (row-wise sum) using NumPy
    total_marks = np.sum(marks_matrix, axis=1)

    # Calculate Average Marks across axis 1 (row-wise mean) using NumPy
    average_marks = np.mean(marks_matrix, axis=1)

    df_result["Total_Marks"] = np.round(total_marks, 2)
    df_result["Average_Marks"] = np.round(average_marks, 2)

    return df_result


# ============================================================================
# 4. GRADING & PASS/FAIL LOGIC
# ============================================================================

def determine_single_student_grade(
    average_mark: float,
    subject_marks: List[float],
    pass_threshold: float = PASS_MARK_THRESHOLD
) -> Tuple[str, str]:
    """
    Evaluates pass/fail status and assigns a letter grade for a single student.

    PROJECT GRADING RULES (Project-defined, not official university rules):
      - Pass Condition: Student must score at least 35 in EVERY subject.
      - If ANY subject mark < 35:
          Status = 'Fail', Grade = 'F'
      - If student passed ALL subjects (every subject >= 35):
          Average >= 90: Grade 'A'
          Average >= 75 and < 90: Grade 'B'
          Average >= 60 and < 75: Grade 'C'
          Average >= 50 and < 60: Grade 'D'
          Average >= 35 and < 50: Grade 'E'
          Average < 35: Grade 'F'

    Parameters:
        average_mark (float): Average score of student.
        subject_marks (list of float): List of marks in all 5 subjects.
        pass_threshold (float): Minimum passing mark per subject (default 35.0).

    Returns:
        tuple: (status, grade) e.g., ('Pass', 'A') or ('Fail', 'F')
    """
    # Check if student cleared every subject
    has_failed_any_subject = any(mark < pass_threshold for mark in subject_marks)

    if has_failed_any_subject:
        return "Fail", "F"

    # If student passed all subjects, determine grade based on average
    if average_mark >= 90.0:
        grade = "A"
    elif average_mark >= 75.0:
        grade = "B"
    elif average_mark >= 60.0:
        grade = "C"
    elif average_mark >= 50.0:
        grade = "D"
    elif average_mark >= 35.0:
        grade = "E"
    else:
        grade = "F"

    return "Pass", grade


def assign_grades_and_status(
    df: pd.DataFrame,
    subject_columns: Optional[List[str]] = None,
    pass_threshold: float = PASS_MARK_THRESHOLD
) -> pd.DataFrame:
    """
    Applies grading and pass/fail rules to all students in the DataFrame.
    Demonstrates Python loops and conditional logic alongside NumPy vectorization.

    Parameters:
        df (pd.DataFrame): DataFrame containing subject marks and Average_Marks.
        subject_columns (list): Names of subject columns.
        pass_threshold (float): Passing cutoff mark per subject.

    Returns:
        pd.DataFrame: DataFrame updated with 'Status' and 'Grade' columns.
    """
    if subject_columns is None:
        subject_columns = DEFAULT_SUBJECTS

    df_result = df.copy()

    statuses = []
    grades = []

    # Iterative loop across each student record to evaluate conditions
    for _, row in df_result.iterrows():
        student_marks = [float(row[sub]) for sub in subject_columns]
        avg = float(row["Average_Marks"])
        status, grade = determine_single_student_grade(avg, student_marks, pass_threshold)
        statuses.append(status)
        grades.append(grade)

    df_result["Status"] = statuses
    df_result["Grade"] = grades

    return df_result


# ============================================================================
# 5. CLASS PERFORMANCE STATISTICS
# ============================================================================

def calculate_class_statistics(
    df: pd.DataFrame,
    subject_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes summary class-wide performance metrics using NumPy.

    Parameters:
        df (pd.DataFrame): DataFrame with metrics, status, and grade.
        subject_columns (list): List of subject column names.

    Returns:
        dict: Dictionary containing class metrics:
              - total_students
              - passed_students
              - failed_students
              - pass_percentage
              - highest_average & student details
              - lowest_average & student details
              - class_mean_average
              - class_std_average
              - grade_counts
    """
    if subject_columns is None:
        subject_columns = DEFAULT_SUBJECTS

    total_students = len(df)
    if total_students == 0:
        raise ValueError("Cannot calculate class statistics: Dataset contains no student records.")

    averages_np = df["Average_Marks"].to_numpy(dtype=float)
    totals_np = df["Total_Marks"].to_numpy(dtype=float)

    # Class performance statistics via NumPy
    highest_avg = np.max(averages_np)
    lowest_avg = np.min(averages_np)
    class_mean = np.mean(averages_np)
    class_std = np.std(averages_np)

    highest_total = np.max(totals_np)
    lowest_total = np.min(totals_np)

    # Identify students holding highest and lowest scores
    top_students = df[df["Average_Marks"] == highest_avg][["Student_ID", "Name", "Average_Marks"]].to_dict(orient="records")
    lowest_students = df[df["Average_Marks"] == lowest_avg][["Student_ID", "Name", "Average_Marks"]].to_dict(orient="records")

    passed_count = int(np.sum(df["Status"] == "Pass"))
    failed_count = total_students - passed_count
    pass_percentage = round((passed_count / total_students) * 100.0, 2) if total_students > 0 else 0.0

    # Grade distribution breakdown
    possible_grades = ["A", "B", "C", "D", "E", "F"]
    grade_distribution = {g: int(np.sum(df["Grade"] == g)) for g in possible_grades}

    return {
        "total_students": total_students,
        "passed_students": passed_count,
        "failed_students": failed_count,
        "pass_percentage": pass_percentage,
        "highest_average": round(float(highest_avg), 2),
        "lowest_average": round(float(lowest_avg), 2),
        "highest_total": round(float(highest_total), 2),
        "lowest_total": round(float(lowest_total), 2),
        "class_mean_average": round(float(class_mean), 2),
        "class_std_average": round(float(class_std), 2),
        "top_students": top_students,
        "lowest_students": lowest_students,
        "grade_distribution": grade_distribution
    }


# ============================================================================
# 6. SUBJECT-WISE PERFORMANCE ANALYSIS
# ============================================================================

def calculate_subject_analysis(
    df: pd.DataFrame,
    subject_columns: Optional[List[str]] = None,
    pass_threshold: float = PASS_MARK_THRESHOLD
) -> pd.DataFrame:
    """
    Performs granular analysis across each subject using NumPy aggregations.

    Calculates:
      - Subject Average Marks
      - Highest Mark Scored
      - Lowest Mark Scored
      - Standard Deviation
      - Subject Pass Rate (%)

    Parameters:
        df (pd.DataFrame): Validated student dataframe.
        subject_columns (list): List of subject column names.
        pass_threshold (float): Cutoff for passing individual subject.

    Returns:
        pd.DataFrame: Table of subject performance metrics.
    """
    if subject_columns is None:
        subject_columns = DEFAULT_SUBJECTS

    records = []
    total_students = len(df)
    if total_students == 0:
        raise ValueError("Cannot calculate subject analysis: Dataset contains no student records.")

    for sub in subject_columns:
        marks_array = df[sub].to_numpy(dtype=float)

        avg_mark = np.mean(marks_array)
        highest_mark = np.max(marks_array)
        lowest_mark = np.min(marks_array)
        std_mark = np.std(marks_array)
        passed_sub_count = np.sum(marks_array >= pass_threshold)
        pass_rate = (passed_sub_count / total_students) * 100.0 if total_students > 0 else 0.0

        records.append({
            "Subject": sub,
            "Average_Marks": round(float(avg_mark), 2),
            "Highest_Mark": round(float(highest_mark), 2),
            "Lowest_Mark": round(float(lowest_mark), 2),
            "Std_Deviation": round(float(std_mark), 2),
            "Pass_Rate_%": round(float(pass_rate), 2)
        })

    subject_df = pd.DataFrame(records)
    return subject_df


# ============================================================================
# 7. TOP PERFORMERS WITH PROPER TIE HANDLING
# ============================================================================

def get_top_performers(
    df: pd.DataFrame,
    top_n: int = 3,
    rank_by: str = "Average_Marks"
) -> pd.DataFrame:
    """
    Identifies top-performing students and correctly handles score ties.

    Tie Handling Strategy:
      Uses dense ranking (`method='dense'`).
      Students who achieve identical scores receive identical rank numbers.
      All students who place in the top N distinct ranks are included,
      ensuring no tied student is unfairly omitted by arbitrary slicing.

    Parameters:
        df (pd.DataFrame): Student dataframe with calculated marks.
        top_n (int): Number of top ranks to include (default: 3).
        rank_by (str): Metric column to rank by (default: 'Average_Marks').

    Returns:
        pd.DataFrame: Top performing students sorted by Rank and Name.
    """
    columns_to_show = ["Rank", "Student_ID", "Name", "Department", "Total_Marks", "Average_Marks", "Grade", "Status"]
    if len(df) == 0:
        return pd.DataFrame(columns=[col for col in columns_to_show if col in df.columns or col == "Rank"])

    df_sorted = df.copy()

    # Calculate dense rank in descending order
    df_sorted["Rank"] = df_sorted[rank_by].rank(method="dense", ascending=False).astype(int)

    # Filter to ranks <= top_n
    top_df = df_sorted[df_sorted["Rank"] <= top_n].sort_values(
        by=["Rank", "Total_Marks", "Name"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    return top_df[[col for col in columns_to_show if col in top_df.columns]]


# ============================================================================
# 8. ATTENDANCE ANALYSIS & LOW ATTENDANCE ALERT
# ============================================================================

def analyze_attendance(
    df: pd.DataFrame,
    threshold: float = ATTENDANCE_WARNING_THRESHOLD
) -> Dict[str, Any]:
    """
    Analyzes student attendance patterns and identifies students with low attendance.

    IMPORTANT NOTE:
      Attendance is evaluated as an institutional compliance criterion and
      DOES NOT determine academic pass/fail status in this analytics model.

    Parameters:
        df (pd.DataFrame): Student dataframe containing 'Attendance' column.
        threshold (float): Minimum attendance threshold percentage (default: 75.0%).

    Returns:
        dict:
          - class_avg_attendance: Mean attendance percentage.
          - highest_attendance: Highest attendance recorded.
          - lowest_attendance: Lowest attendance recorded.
          - threshold: Threshold percentage applied.
          - low_attendance_count: Number of students below threshold.
          - low_attendance_students: DataFrame of students below threshold.
          - compliant_percentage: Percentage of students with attendance >= threshold.
    """
    if len(df) == 0:
        raise ValueError("Cannot analyze attendance: Dataset contains no student records.")
    attendance_np = df["Attendance"].to_numpy(dtype=float)

    avg_att = np.mean(attendance_np)
    max_att = np.max(attendance_np)
    min_att = np.min(attendance_np)

    low_att_mask = attendance_np < threshold
    low_att_count = int(np.sum(low_att_mask))
    total_students = len(df)
    compliant_pct = round(((total_students - low_att_count) / total_students) * 100.0, 2) if total_students > 0 else 0.0

    low_att_df = df[df["Attendance"] < threshold][
        ["Student_ID", "Name", "Department", "Attendance", "Average_Marks", "Status"]
    ].sort_values(by="Attendance", ascending=True).reset_index(drop=True)

    return {
        "class_avg_attendance": round(float(avg_att), 2),
        "highest_attendance": round(float(max_att), 2),
        "lowest_attendance": round(float(min_att), 2),
        "threshold": threshold,
        "low_attendance_count": low_att_count,
        "low_attendance_students": low_att_df,
        "compliant_percentage": compliant_pct
    }


# ============================================================================
# 9. KEY OBSERVATIONS GENERATOR
# ============================================================================

def generate_key_observations(
    df: pd.DataFrame,
    class_stats: Dict[str, Any],
    subject_analysis_df: pd.DataFrame,
    attendance_summary: Dict[str, Any]
) -> List[str]:
    """
    Generates dynamic, data-driven observations from actual computed metrics.
    Ensures zero fake claims or hardcoded assumptions.

    Parameters:
        df (pd.DataFrame): Fully calculated student DataFrame.
        class_stats (dict): Class statistics dictionary.
        subject_analysis_df (pd.DataFrame): Subject-level statistics DataFrame.
        attendance_summary (dict): Attendance analysis dictionary.

    Returns:
        list of str: Concrete analytical observations.
    """
    observations = []

    # 1. Academic Performance Summary
    observations.append(
        f"Overall Class Performance: {class_stats['passed_students']} out of {class_stats['total_students']} "
        f"students ({class_stats['pass_percentage']}%) cleared all academic requirements."
    )

    # 2. Subject Performance Highlights
    sorted_subjects = subject_analysis_df.sort_values(by="Average_Marks", ascending=False)
    highest_sub = sorted_subjects.iloc[0]
    lowest_sub = sorted_subjects.iloc[-1]

    observations.append(
        f"Strongest Subject: '{highest_sub['Subject']}' recorded the highest average of "
        f"{highest_sub['Average_Marks']} marks with a {highest_sub['Pass_Rate_%']}% pass rate."
    )
    observations.append(
        f"Most Challenging Subject: '{lowest_sub['Subject']}' recorded the lowest average of "
        f"{lowest_sub['Average_Marks']} marks with a {lowest_sub['Pass_Rate_%']}% pass rate."
    )

    # 3. Grading Rule Impact (Students failing due to single subject cutoff)
    failed_df = df[df["Status"] == "Fail"]
    single_sub_failures = []
    for _, row in failed_df.iterrows():
        marks = [row[sub] for sub in DEFAULT_SUBJECTS]
        subs_below_35 = sum(1 for m in marks if m < PASS_MARK_THRESHOLD)
        if subs_below_35 == 1 and row["Average_Marks"] >= 50.0:
            single_sub_failures.append((row["Name"], row["Average_Marks"]))

    if single_sub_failures:
        names_str = ", ".join([f"{name} (Avg: {avg}%)" for name, avg in single_sub_failures])
        observations.append(
            f"Impact of Subject Minimum Rule (35 marks): {names_str} achieved solid overall averages "
            f"but received Grade F due to falling below 35 in a single subject, demonstrating strict policy compliance."
        )

    # 4. Attendance Impact & Correlation
    high_att_students = df[df["Attendance"] >= 75.0]
    low_att_students = df[df["Attendance"] < 75.0]

    avg_high_att = round(float(np.mean(high_att_students["Average_Marks"].values)), 2) if len(high_att_students) > 0 else 0
    avg_low_att = round(float(np.mean(low_att_students["Average_Marks"].values)), 2) if len(low_att_students) > 0 else 0

    observations.append(
        f"Attendance vs Academics Correlation: Students with >= 75% attendance averaged {avg_high_att} marks, "
        f"compared to {avg_low_att} marks for students with low attendance (< 75%), "
        f"showing a {round(avg_high_att - avg_low_att, 2)} mark performance advantage for regular attendees."
    )

    # 5. Low Attendance Alert
    low_att_count = attendance_summary["low_attendance_count"]
    observations.append(
        f"Institutional Compliance: {low_att_count} student(s) ({round(100 - attendance_summary['compliant_percentage'], 2)}%) "
        f"breached the 75% attendance threshold and require institutional counseling."
    )

    return observations


# ============================================================================
# 10. FORMATTED DISPLAY HELPERS
# ============================================================================

def print_section_header(title: str, symbol: str = "=", width: int = 80) -> None:
    """Prints a styled terminal header banner."""
    print("\n" + symbol * width)
    print(f" {title.upper()} ".center(width, symbol))
    print(symbol * width)


def format_table(df: pd.DataFrame, table_format: str = "grid") -> str:
    """Formats a pandas DataFrame into a clean ASCII table."""
    try:
        return tabulate(df, headers="keys", tablefmt=table_format, showindex=False)
    except Exception:
        return df.to_string(index=False)
