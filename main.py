"""
Student Performance Analytics System
====================================
Main execution script for the EWB Courses Python with AI project assignment.

Demonstrates:
  - Dataset loading and validation with Pandas.
  - Vectorized numerical calculations with NumPy.
  - Functional design, loops, and conditional grading rules.
  - Class statistics, subject-wise analytics, tie-aware ranking, and attendance auditing.

Usage:
  python main.py
"""

import sys
import os
import pandas as pd
from tabulate import tabulate

# Ensure UTF-8 console output encoding on Windows systems
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from functions import (
    DEFAULT_SUBJECTS,
    DEFAULT_REQUIRED_COLUMNS,
    PASS_MARK_THRESHOLD,
    ATTENDANCE_WARNING_THRESHOLD,
    load_student_dataset,
    validate_student_data,
    calculate_student_metrics,
    assign_grades_and_status,
    calculate_class_statistics,
    calculate_subject_analysis,
    get_top_performers,
    analyze_attendance,
    generate_key_observations,
    print_section_header,
    format_table,
)


def run_analytics_system(csv_path: str = "students.csv") -> None:
    """
    Orchestrates the end-to-end data processing and analytics pipeline.
    """
    print_section_header("Student Performance Analytics System", symbol="*")
    print("Project : EWB Courses - Python with AI Assignment")
    print("Tech    : Python 3 | NumPy | Pandas | Tabulate")
    print("Rules   : Project-Defined Grading Rules (Academic Model)")
    print("-" * 80)

    # ------------------------------------------------------------------------
    # STEP 1: LOAD DATASET
    # ------------------------------------------------------------------------
    print("\n[Step 1/7] Loading dataset from CSV file...")
    try:
        raw_df = load_student_dataset(csv_path)
        print(f"-> Successfully loaded {len(raw_df)} student records from '{csv_path}'.")
    except Exception as e:
        print(f"CRITICAL ERROR loading dataset: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # STEP 2: DATA VALIDATION
    # ------------------------------------------------------------------------
    print("\n[Step 2/7] Validating data integrity...")
    try:
        val_result = validate_student_data(raw_df, raise_on_error=True)
        print(f"-> Validation passed: 0 missing columns, 0 null cells, 0 duplicate IDs, 0 out-of-range marks.")
    except ValueError as ve:
        print(f"VALIDATION FAILURE: {ve}")
        sys.exit(1)

    # Display Dataset Overview
    print_section_header("1. Dataset Overview (First 5 Fictional Records)", symbol="=")
    overview_cols = ["Student_ID", "Name", "Department"] + DEFAULT_SUBJECTS + ["Attendance"]
    print(format_table(raw_df[overview_cols].head(5), table_format="grid"))
    print(f"\nTotal Records: {len(raw_df)} | Total Attributes: {len(raw_df.columns)}")
    print("Subjects Evaluated (out of 100): " + ", ".join(DEFAULT_SUBJECTS))

    # ------------------------------------------------------------------------
    # STEP 3: CALCULATE METRICS & ASSIGN GRADES
    # ------------------------------------------------------------------------
    print("\n[Step 3/7] Calculating totals, averages, and applying grading rules...")
    # Calculate Total Marks and Average Marks using NumPy
    metrics_df = calculate_student_metrics(raw_df, subject_columns=DEFAULT_SUBJECTS)

    # Assign Status (Pass/Fail) and Grade (A, B, C, D, E, F)
    processed_df = assign_grades_and_status(metrics_df, subject_columns=DEFAULT_SUBJECTS)

    print_section_header("2. Student Results (Totals, Averages, Grades & Status)", symbol="=")
    result_display_cols = [
        "Student_ID", "Name", "Department",
        "Total_Marks", "Average_Marks", "Grade", "Status", "Attendance"
    ]
    print(format_table(processed_df[result_display_cols], table_format="grid"))

    print("\nPROJECT-DEFINED GRADING RULES EXPLANATION:")
    print("  * Note: These are project-defined assignment rules, not official university rules.")
    print(f"  * Pass Condition: Minimum {PASS_MARK_THRESHOLD:.0f} marks required in EVERY subject.")
    print("  * Fail Rule     : Any subject < 35 marks results in Grade 'F' and Status 'Fail'.")
    print("  * Grade Scale   : A (>=90%), B (75-89%), C (60-74%), D (50-59%), E (35-49%), F (<35% or subject fail).")
    print("  * Attendance    : Tracked separately as institutional compliance (does not alter academic grade).")

    # ------------------------------------------------------------------------
    # STEP 4: CLASS-WIDE STATISTICS
    # ------------------------------------------------------------------------
    print("\n[Step 4/7] Generating class-wide performance metrics...")
    class_stats = calculate_class_statistics(processed_df, subject_columns=DEFAULT_SUBJECTS)

    # Format names of highest and lowest performing students
    top_student_names = ", ".join([f"{s['Name']} ({s['Student_ID']})" for s in class_stats["top_students"]])
    lowest_student_names = ", ".join([f"{s['Name']} ({s['Student_ID']})" for s in class_stats["lowest_students"]])

    print_section_header("3. Class Performance Statistics", symbol="=")
    stats_table_data = [
        ["Total Enrolled Students", class_stats["total_students"]],
        ["Students Passed", f"{class_stats['passed_students']} ({class_stats['pass_percentage']}%)"],
        ["Students Failed", f"{class_stats['failed_students']} ({round(100 - class_stats['pass_percentage'], 2)}%)"],
        ["Highest-Performing Student(s)", f"{top_student_names} [{class_stats['highest_average']}%]"],
        ["Lowest-Performing Student(s)", f"{lowest_student_names} [{class_stats['lowest_average']}%]"],
        ["Highest Total Marks", f"{class_stats['highest_total']} / 500.0"],
        ["Lowest Total Marks", f"{class_stats['lowest_total']} / 500.0"],
        ["Highest Average Mark", f"{class_stats['highest_average']}%"],
        ["Lowest Average Mark", f"{class_stats['lowest_average']}%"],
        ["Class Overall Mean Average", f"{class_stats['class_mean_average']}%"],
        ["Class Standard Deviation", f"{class_stats['class_std_average']}%"],
    ]
    print(tabulate(stats_table_data, headers=["Metric", "Value"], tablefmt="grid"))

    # Grade distribution breakdown
    print("\nGrade Distribution:")
    grade_rows = [[g, count, f"{(count / class_stats['total_students']) * 100:.1f}%"]
                  for g, count in class_stats["grade_distribution"].items()]
    print(tabulate(grade_rows, headers=["Grade", "Student Count", "Percentage"], tablefmt="simple"))

    # ------------------------------------------------------------------------
    # STEP 5: SUBJECT-WISE PERFORMANCE ANALYSIS
    # ------------------------------------------------------------------------
    print("\n[Step 5/7] Analyzing subject-wise performance...")
    subject_analysis = calculate_subject_analysis(processed_df, subject_columns=DEFAULT_SUBJECTS)

    print_section_header("4. Subject-Wise Performance Analysis", symbol="=")
    print(format_table(subject_analysis, table_format="grid"))

    # Explicitly highlight best-performing and most challenging subjects
    best_sub = subject_analysis.sort_values(by="Average_Marks", ascending=False).iloc[0]
    worst_sub = subject_analysis.sort_values(by="Average_Marks", ascending=True).iloc[0]
    print(f"\n* Best-Performing Subject : '{best_sub['Subject']}' (Class Mean: {best_sub['Average_Marks']}%, Pass Rate: {best_sub['Pass_Rate_%']}%)")
    print(f"* Most Challenging Subject: '{worst_sub['Subject']}' (Class Mean: {worst_sub['Average_Marks']}%, Pass Rate: {worst_sub['Pass_Rate_%']}%)")

    # ------------------------------------------------------------------------
    # STEP 6: TOP PERFORMERS (WITH TIE-HANDLING)
    # ------------------------------------------------------------------------
    print("\n[Step 6/7] Identifying top performers with tie-aware ranking...")
    top_performers = get_top_performers(processed_df, top_n=3, rank_by="Average_Marks")

    print_section_header("5. Top Performing Students (Top 3 Ranks with Ties)", symbol="=")
    print(format_table(top_performers, table_format="grid"))
    print("\nTie-Handling Note:")
    print("  Uses dense ranking: Students with identical average scores share the exact same rank.")
    print("  All students in the top 3 score tiers are included without unfair truncation.")

    # ------------------------------------------------------------------------
    # STEP 7: ATTENDANCE ANALYSIS & OBSERVATIONS
    # ------------------------------------------------------------------------
    print("\n[Step 7/7] Analyzing attendance records and generating key insights...")
    attendance_summary = analyze_attendance(processed_df, threshold=ATTENDANCE_WARNING_THRESHOLD)

    print_section_header("6. Attendance Summary & Low Attendance Alert", symbol="=")
    att_meta_data = [
        ["Class Average Attendance", f"{attendance_summary['class_avg_attendance']}%"],
        ["Highest Attendance", f"{attendance_summary['highest_attendance']}%"],
        ["Lowest Attendance", f"{attendance_summary['lowest_attendance']}%"],
        ["Compliance Threshold", f"{attendance_summary['threshold']}%"],
        ["Compliant Students (>= 75%)", f"{class_stats['total_students'] - attendance_summary['low_attendance_count']} ({attendance_summary['compliant_percentage']}%)"],
        ["Students Below Threshold (< 75%)", f"{attendance_summary['low_attendance_count']} ({round(100 - attendance_summary['compliant_percentage'], 2)}%)"]
    ]
    print(tabulate(att_meta_data, headers=["Attendance Indicator", "Value"], tablefmt="grid"))

    if attendance_summary["low_attendance_count"] > 0:
        print(f"\n[!] WARNING: The following {attendance_summary['low_attendance_count']} student(s) have fallen below the 75% attendance threshold:")
        print(format_table(attendance_summary["low_attendance_students"], table_format="grid"))
    else:
        print("\nAll students satisfy the minimum attendance threshold (>= 75%).")

    # Key Observations
    observations = generate_key_observations(processed_df, class_stats, subject_analysis, attendance_summary)

    print_section_header("7. Key Analytical Observations (Data-Driven Insights)", symbol="=")
    for idx, obs in enumerate(observations, start=1):
        print(f"  {idx}. {obs}")

    print("\n" + "=" * 80)
    print(" Analytics processing complete. All outputs generated successfully! ".center(80, "#"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Locate dataset in current workspace directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, "students.csv")
    run_analytics_system(dataset_file)
