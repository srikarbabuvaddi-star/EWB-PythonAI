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


# ============================================================================
# 11. WEB DASHBOARD GENERATOR (VERCEL & BROWSER DEPLOYMENT)
# ============================================================================

def generate_html_dashboard(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    class_stats: Dict[str, Any],
    subject_analysis: pd.DataFrame,
    top_performers: pd.DataFrame,
    attendance_summary: Dict[str, Any],
    observations: List[str]
) -> str:
    """
    Renders a responsive, modern HTML dashboard of the analytics report
    for browser deployment on Vercel or cloud web servers.
    """
    # Grade distribution rows
    grade_distribution_html = "".join([
        f'<div class="grade-pill"><span class="grade-tag grade-{g}">{g}</span>'
        f'<span class="grade-count">{count}</span>'
        f'<span class="grade-pct">({(count / class_stats["total_students"]) * 100:.1f}%)</span></div>'
        for g, count in class_stats["grade_distribution"].items()
    ])

    # Student results rows
    student_rows_html = []
    for _, row in processed_df.iterrows():
        grade_badge = f'<span class="badge grade-{row["Grade"]}">{row["Grade"]}</span>'
        status_badge = f'<span class="badge status-{row["Status"].lower()}">{row["Status"]}</span>'
        att_style = 'style="color: #ef4444; font-weight: 600;"' if row["Attendance"] < 75.0 else ""
        student_rows_html.append(
            f'<tr>'
            f'<td><strong>{row["Student_ID"]}</strong></td>'
            f'<td>{row["Name"]}</td>'
            f'<td>{row["Department"]}</td>'
            f'<td>{row["Total_Marks"]} / 500</td>'
            f'<td>{row["Average_Marks"]}%</td>'
            f'<td>{grade_badge}</td>'
            f'<td>{status_badge}</td>'
            f'<td {att_style}>{row["Attendance"]}%</td>'
            f'</tr>'
        )
    students_table_body = "\n".join(student_rows_html)

    # Subject analysis rows
    best_sub = subject_analysis.sort_values(by="Average_Marks", ascending=False).iloc[0]["Subject"]
    worst_sub = subject_analysis.sort_values(by="Average_Marks", ascending=True).iloc[0]["Subject"]

    subject_rows_html = []
    for _, row in subject_analysis.iterrows():
        sub_name = row["Subject"]
        callout = ""
        if sub_name == best_sub:
            callout = ' <span class="badge badge-success">Top Subject</span>'
        elif sub_name == worst_sub:
            callout = ' <span class="badge badge-warning">Most Challenging</span>'

        subject_rows_html.append(
            f'<tr>'
            f'<td><strong>{sub_name}</strong>{callout}</td>'
            f'<td>{row["Average_Marks"]}%</td>'
            f'<td>{row["Highest_Mark"]}</td>'
            f'<td>{row["Lowest_Mark"]}</td>'
            f'<td>{row["Std_Deviation"]}</td>'
            f'<td>{row["Pass_Rate_%"]}%</td>'
            f'</tr>'
        )
    subjects_table_body = "\n".join(subject_rows_html)

    # Top performers rows
    top_rows_html = []
    for _, row in top_performers.iterrows():
        top_rows_html.append(
            f'<tr>'
            f'<td><span class="rank-circle">#{row["Rank"]}</span></td>'
            f'<td><strong>{row["Name"]}</strong> ({row["Student_ID"]})</td>'
            f'<td>{row["Department"]}</td>'
            f'<td>{row["Total_Marks"]}</td>'
            f'<td><strong>{row["Average_Marks"]}%</strong></td>'
            f'<td><span class="badge grade-{row["Grade"]}">{row["Grade"]}</span></td>'
            f'<td><span class="badge status-pass">Pass</span></td>'
            f'</tr>'
        )
    top_table_body = "\n".join(top_rows_html)

    # Low attendance rows
    low_att_rows_html = []
    for _, row in attendance_summary["low_attendance_students"].iterrows():
        low_att_rows_html.append(
            f'<tr>'
            f'<td><strong>{row["Student_ID"]}</strong></td>'
            f'<td>{row["Name"]}</td>'
            f'<td>{row["Department"]}</td>'
            f'<td style="color: #ef4444; font-weight: 700;">{row["Attendance"]}%</td>'
            f'<td>{row["Average_Marks"]}%</td>'
            f'<td><span class="badge status-{row["Status"].lower()}">{row["Status"]}</span></td>'
            f'</tr>'
        )
    low_att_table_body = "\n".join(low_att_rows_html)

    # Observations list
    observations_html = "\n".join([f'<li>{obs}</li>' for obs in observations])

    # Names of top & lowest students
    top_names = ", ".join([f"{s['Name']} ({s['Student_ID']})" for s in class_stats['top_students']])
    lowest_names = ", ".join([f"{s['Name']} ({s['Student_ID']})" for s in class_stats['lowest_students']])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Student Performance Analytics System</title>
  <style>
    :root {{
      --bg-color: #0f172a;
      --card-bg: #1e293b;
      --border-color: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-color);
      color: var(--text-main);
      font-family: var(--font-family);
      line-height: 1.6;
      padding: 2rem 1rem;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      text-align: center;
      margin-bottom: 2.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border-color);
    }}
    h1 {{
      font-size: 2.2rem;
      color: #ffffff;
      margin-bottom: 0.5rem;
      letter-spacing: -0.5px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 1.05rem;
    }}
    .repo-link {{
      display: inline-block;
      margin-top: 0.75rem;
      color: var(--primary);
      text-decoration: none;
      font-weight: 500;
    }}
    .repo-link:hover {{ text-decoration: underline; }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.25rem;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stat-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}
    .stat-label {{
      color: var(--text-muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 0.4rem;
    }}
    .stat-value {{
      font-size: 1.8rem;
      font-weight: 700;
      color: #ffffff;
    }}
    .stat-sub {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }}
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      overflow-x: auto;
    }}
    .section-title {{
      font-size: 1.25rem;
      color: #ffffff;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .badge-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
    .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
    .status-pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .status-fail {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
    .grade-A {{ background: #065f46; color: #6ee7b7; }}
    .grade-B {{ background: #1e40af; color: #93c5fd; }}
    .grade-C {{ background: #854d0e; color: #fde047; }}
    .grade-D {{ background: #9a3412; color: #fdba74; }}
    .grade-E {{ background: #78350f; color: #fcd34d; }}
    .grade-F {{ background: #991b1b; color: #fca5a5; }}
    .grades-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-top: 1rem;
    }}
    .grade-pill {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
      background: #0f172a;
      padding: 0.4rem 0.8rem;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      font-size: 0.85rem;
    }}
    .grade-tag {{
      display: inline-block;
      width: 20px;
      height: 20px;
      line-height: 20px;
      text-align: center;
      border-radius: 4px;
      font-weight: 700;
      font-size: 0.75rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      text-align: left;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border-color);
    }}
    th {{
      color: var(--text-muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      background: #172033;
    }}
    tbody tr:hover {{
      background: rgba(255, 255, 255, 0.03);
    }}
    .rank-circle {{
      display: inline-block;
      width: 28px;
      height: 28px;
      line-height: 28px;
      text-align: center;
      border-radius: 50%;
      background: #38bdf8;
      color: #0f172a;
      font-weight: 800;
      font-size: 0.85rem;
    }}
    .alert-box {{
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
      color: #fca5a5;
      font-size: 0.9rem;
    }}
    ul.observations-list {{
      padding-left: 1.5rem;
    }}
    ul.observations-list li {{
      margin-bottom: 0.75rem;
      color: #cbd5e1;
    }}
    footer {{
      text-align: center;
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border-color);
      color: var(--text-muted);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Student Performance Analytics System</h1>
      <p class="subtitle">EWB Courses: Python with AI Practical Assignment | Evaluated with Python, NumPy & Pandas</p>
      <a class="repo-link" href="https://github.com/srikarbabuvaddi-star/EWB-PythonAI" target="_blank">View GitHub Repository &rarr;</a>
    </header>

    <!-- Top Key Metrics -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Total Students</div>
        <div class="stat-value">{class_stats["total_students"]}</div>
        <div class="stat-sub">Across 4 Departments</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Pass Percentage</div>
        <div class="stat-value" style="color: #34d399;">{class_stats["pass_percentage"]}%</div>
        <div class="stat-sub">{class_stats["passed_students"]} Passed / {class_stats["failed_students"]} Failed</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Class Mean Average</div>
        <div class="stat-value">{class_stats["class_mean_average"]}%</div>
        <div class="stat-sub">Std Dev: {class_stats["class_std_average"]}%</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Highest Score</div>
        <div class="stat-value" style="color: #38bdf8;">{class_stats["highest_average"]}%</div>
        <div class="stat-sub">{top_names}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Attendance Compliance</div>
        <div class="stat-value">{attendance_summary["compliant_percentage"]}%</div>
        <div class="stat-sub">{attendance_summary["low_attendance_count"]} Warning Alert(s)</div>
      </div>
    </div>

    <!-- Section 1: Complete Results Table -->
    <div class="section-card">
      <h2 class="section-title">1. Student Performance Table</h2>
      <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
        Grading Policy: Minimum 35 marks in every subject is mandatory to pass. Attendance is monitored separately as institutional compliance.
      </p>
      <table>
        <thead>
          <tr>
            <th>Student ID</th>
            <th>Name</th>
            <th>Department</th>
            <th>Total Marks</th>
            <th>Average Marks</th>
            <th>Grade</th>
            <th>Status</th>
            <th>Attendance</th>
          </tr>
        </thead>
        <tbody>
          {students_table_body}
        </tbody>
      </table>
    </div>

    <!-- Section 2: Class Performance & Grade Distribution -->
    <div class="section-card">
      <h2 class="section-title">2. Class Performance & Grade Distribution</h2>
      <p style="color: var(--text-muted); font-size: 0.85rem;">
        Grades: A (&ge;90%), B (75-89%), C (60-74%), D (50-59%), E (35-49%), F (&lt;35% or any subject fail).
      </p>
      <div class="grades-bar">
        {grade_distribution_html}
      </div>
    </div>

    <!-- Section 3: Subject-Wise Performance -->
    <div class="section-card">
      <h2 class="section-title">3. Subject-Wise Performance Analysis</h2>
      <table>
        <thead>
          <tr>
            <th>Subject</th>
            <th>Class Average</th>
            <th>Highest Mark</th>
            <th>Lowest Mark</th>
            <th>Std. Deviation</th>
            <th>Pass Rate</th>
          </tr>
        </thead>
        <tbody>
          {subjects_table_body}
        </tbody>
      </table>
    </div>

    <!-- Section 4: Top Performers (Dense Rank Ties) -->
    <div class="section-card">
      <h2 class="section-title">4. Top Performing Students (Dense Ranking with Ties)</h2>
      <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem;">
        All students achieving top 3 score tiers are included without unfair truncation.
      </p>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Student</th>
            <th>Department</th>
            <th>Total Marks</th>
            <th>Average</th>
            <th>Grade</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {top_table_body}
        </tbody>
      </table>
    </div>

    <!-- Section 5: Attendance Alerts -->
    <div class="section-card">
      <h2 class="section-title">5. Attendance Compliance & Warning Alerts (&lt; 75%)</h2>
      <div class="alert-box">
        <strong>Notice:</strong> The following {attendance_summary["low_attendance_count"]} student(s) have fallen below the mandatory 75% institutional attendance threshold and require academic counseling.
      </div>
      <table>
        <thead>
          <tr>
            <th>Student ID</th>
            <th>Name</th>
            <th>Department</th>
            <th>Attendance</th>
            <th>Average Score</th>
            <th>Academic Status</th>
          </tr>
        </thead>
        <tbody>
          {low_att_table_body}
        </tbody>
      </table>
    </div>

    <!-- Section 6: Key Observations -->
    <div class="section-card">
      <h2 class="section-title">6. Key Analytical Observations</h2>
      <ul class="observations-list">
        {observations_html}
      </ul>
    </div>

    <footer>
      Student Performance Analytics System &bull; EWB Courses Python with AI Assignment &bull; Built with Python, NumPy & Pandas
    </footer>
  </div>
</body>
</html>"""
    return html_content

