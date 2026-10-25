"""
Student Performance Analytics System - Unit and Integration Tests
=================================================================
Automated test suite validating all calculation, grading, validation,
ranking, and attendance logic using PyTest.

Run with:
  pytest test_analytics.py -v
"""

import pytest
import numpy as np
import pandas as pd

from functions import (
    DEFAULT_SUBJECTS,
    PASS_MARK_THRESHOLD,
    validate_student_data,
    calculate_student_metrics,
    determine_single_student_grade,
    assign_grades_and_status,
    calculate_class_statistics,
    calculate_subject_analysis,
    get_top_performers,
    analyze_attendance,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_valid_df():
    """Returns a valid 5-student sample DataFrame."""
    return pd.DataFrame({
        "Student_ID": ["S01", "S02", "S03", "S04", "S05"],
        "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "Department": ["CS", "AI", "DS", "IT", "CS"],
        "Python": [95, 80, 65, 50, 40],
        "Data_Structures": [90, 75, 60, 52, 38],
        "DBMS": [92, 78, 62, 54, 42],
        "Statistics": [94, 82, 68, 51, 36],
        "AI_Basics": [96, 85, 70, 53, 39],
        "Attendance": [92.0, 85.0, 74.0, 68.0, 80.0]
    })


@pytest.fixture
def sample_tie_df():
    """Returns a DataFrame designed with tied scores for top performer testing."""
    return pd.DataFrame({
        "Student_ID": ["T01", "T02", "T03", "T04", "T05"],
        "Name": ["Student One", "Student Two", "Student Three", "Student Four", "Student Five"],
        "Department": ["CS", "AI", "DS", "IT", "CS"],
        "Python": [95, 95, 85, 85, 70],
        "Data_Structures": [95, 95, 85, 85, 70],
        "DBMS": [95, 95, 85, 85, 70],
        "Statistics": [95, 95, 85, 85, 70],
        "AI_Basics": [95, 95, 85, 85, 70],
        "Attendance": [90.0, 91.0, 85.0, 84.0, 80.0]
    })


# ============================================================================
# 1. TOTAL AND AVERAGE CALCULATIONS
# ============================================================================

def test_calculate_student_metrics(sample_valid_df):
    """Verify that NumPy calculations for row sums and means match expected arithmetic."""
    result_df = calculate_student_metrics(sample_valid_df)

    # First student: 95 + 90 + 92 + 94 + 96 = 467; Average = 93.4
    assert result_df.loc[0, "Total_Marks"] == 467.0
    assert result_df.loc[0, "Average_Marks"] == 93.4

    # Fifth student: 40 + 38 + 42 + 36 + 39 = 195; Average = 39.0
    assert result_df.loc[4, "Total_Marks"] == 195.0
    assert result_df.loc[4, "Average_Marks"] == 39.0

    # Ensure output is a DataFrame and columns exist
    assert "Total_Marks" in result_df.columns
    assert "Average_Marks" in result_df.columns


# ============================================================================
# 2. GRADE BOUNDARY TESTS
# ============================================================================

@pytest.mark.parametrize("average, marks, expected_status, expected_grade", [
    # Grade A (>= 90)
    (90.0, [90, 90, 90, 90, 90], "Pass", "A"),
    (95.5, [95, 96, 94, 98, 95], "Pass", "A"),
    (89.99, [90, 90, 90, 90, 89], "Pass", "B"),

    # Grade B (>= 75 and < 90)
    (75.0, [75, 75, 75, 75, 75], "Pass", "B"),
    (85.0, [85, 85, 85, 85, 85], "Pass", "B"),
    (74.99, [75, 75, 75, 75, 74], "Pass", "C"),

    # Grade C (>= 60 and < 75)
    (60.0, [60, 60, 60, 60, 60], "Pass", "C"),
    (68.5, [70, 65, 70, 68, 70], "Pass", "C"),
    (59.99, [60, 60, 60, 60, 59], "Pass", "D"),

    # Grade D (>= 50 and < 60)
    (50.0, [50, 50, 50, 50, 50], "Pass", "D"),
    (55.0, [55, 55, 55, 55, 55], "Pass", "D"),
    (49.99, [50, 50, 50, 50, 49], "Pass", "E"),

    # Grade E (>= 35 and < 50)
    (35.0, [35, 35, 35, 35, 35], "Pass", "E"),
    (45.0, [45, 45, 45, 45, 45], "Pass", "E"),
    (34.99, [35, 35, 35, 35, 34], "Fail", "F"),  # 34 is < 35, so Fail
])
def test_grade_boundaries(average, marks, expected_status, expected_grade):
    """Test precise boundary conditions for all letter grades."""
    status, grade = determine_single_student_grade(average, marks)
    assert status == expected_status
    assert grade == expected_grade


# ============================================================================
# 3. PASS / FAIL LOGIC
# ============================================================================

def test_pass_condition_all_above_threshold():
    """Student with all marks >= 35 passes."""
    marks = [35, 36, 40, 50, 60]
    avg = np.mean(marks)
    status, grade = determine_single_student_grade(avg, marks)
    assert status == "Pass"
    assert grade != "F"


def test_fail_condition_single_subject_below_threshold():
    """Even with a very high total/average, scoring below 35 in ONE subject must trigger Fail and Grade F."""
    marks = [95, 92, 90, 88, 34]  # 34 in 5th subject (< 35)
    avg = np.mean(marks)  # 79.8 (would otherwise be B)
    status, grade = determine_single_student_grade(avg, marks)
    assert status == "Fail"
    assert grade == "F"


def test_assign_grades_and_status_dataframe():
    """Verify batch grade assignment on DataFrame."""
    df = pd.DataFrame({
        "Student_ID": ["S1", "S2"],
        "Python": [90, 85],
        "Data_Structures": [90, 85],
        "DBMS": [90, 85],
        "Statistics": [90, 30],  # S2 fails statistics
        "AI_Basics": [90, 85],
        "Average_Marks": [90.0, 74.0]
    })
    result = assign_grades_and_status(df)
    assert result.loc[0, "Status"] == "Pass"
    assert result.loc[0, "Grade"] == "A"
    assert result.loc[1, "Status"] == "Fail"
    assert result.loc[1, "Grade"] == "F"


# ============================================================================
# 4. SUBJECT-WISE PERFORMANCE ANALYSIS
# ============================================================================

def test_calculate_subject_analysis(sample_valid_df):
    """Verify that subject analytics calculate correct min, max, mean, and pass rate."""
    subj_df = calculate_subject_analysis(sample_valid_df)

    # 5 subjects evaluated
    assert len(subj_df) == 5

    # Check Python column: [95, 80, 65, 50, 40]
    python_stats = subj_df[subj_df["Subject"] == "Python"].iloc[0]
    assert python_stats["Highest_Mark"] == 95.0
    assert python_stats["Lowest_Mark"] == 40.0
    assert python_stats["Average_Marks"] == 66.0  # (95+80+65+50+40)/5 = 66.0
    assert python_stats["Pass_Rate_%"] == 100.0   # All >= 35


# ============================================================================
# 5. TOP PERFORMERS & TIE HANDLING
# ============================================================================

def test_top_performers_and_ties(sample_tie_df):
    """
    Test tie handling logic:
      T01 and T02 both have Average = 95.0 (Tied for Rank 1)
      T03 and T04 both have Average = 85.0 (Tied for Rank 2)
      T05 has Average = 70.0 (Rank 3)
    Requesting top_n=2 must return 4 students (2 at Rank 1, 2 at Rank 2).
    """
    metrics_df = calculate_student_metrics(sample_tie_df)
    graded_df = assign_grades_and_status(metrics_df)

    top_2_df = get_top_performers(graded_df, top_n=2)

    # With dense ranking, both Rank 1 students and both Rank 2 students must be included
    assert len(top_2_df) == 4
    rank_1_students = top_2_df[top_2_df["Rank"] == 1]
    rank_2_students = top_2_df[top_2_df["Rank"] == 2]

    assert len(rank_1_students) == 2
    assert set(rank_1_students["Student_ID"]) == {"T01", "T02"}
    assert len(rank_2_students) == 2
    assert set(rank_2_students["Student_ID"]) == {"T03", "T04"}


# ============================================================================
# 6. DATA VALIDATION: INVALID MARKS, MISSING DATA, DUPLICATES
# ============================================================================

def test_validation_missing_required_column():
    """Validation must fail when a required column is missing."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John"],
        "Department": ["CS"]
        # Missing subjects and Attendance
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_student_data(df, raise_on_error=True)


def test_validation_missing_null_values():
    """Validation must fail when missing/NaN values exist."""
    df = pd.DataFrame({
        "Student_ID": ["S01", "S02"],
        "Name": ["John", "Jane"],
        "Department": ["CS", "AI"],
        "Python": [85, None],  # Null mark
        "Data_Structures": [80, 82],
        "DBMS": [75, 78],
        "Statistics": [70, 72],
        "AI_Basics": [88, 86],
        "Attendance": [85.0, 90.0]
    })
    with pytest.raises(ValueError, match="missing/null values"):
        validate_student_data(df, raise_on_error=True)


def test_validation_duplicate_student_ids():
    """Validation must fail when duplicate Student_IDs are found."""
    df = pd.DataFrame({
        "Student_ID": ["S01", "S01"],  # Duplicate ID
        "Name": ["John", "Jane"],
        "Department": ["CS", "AI"],
        "Python": [85, 82],
        "Data_Structures": [80, 82],
        "DBMS": [75, 78],
        "Statistics": [70, 72],
        "AI_Basics": [88, 86],
        "Attendance": [85.0, 90.0]
    })
    with pytest.raises(ValueError, match="Duplicate Student IDs"):
        validate_student_data(df, raise_on_error=True)


def test_validation_invalid_marks_negative():
    """Validation must fail if marks are negative (< 0)."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John"],
        "Department": ["CS"],
        "Python": [-5],  # Invalid mark
        "Data_Structures": [80],
        "DBMS": [75],
        "Statistics": [70],
        "AI_Basics": [88],
        "Attendance": [85.0]
    })
    with pytest.raises(ValueError, match="marks outside"):
        validate_student_data(df, raise_on_error=True)


def test_validation_invalid_marks_above_100():
    """Validation must fail if marks exceed 100."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John"],
        "Department": ["CS"],
        "Python": [105],  # Invalid mark > 100
        "Data_Structures": [80],
        "DBMS": [75],
        "Statistics": [70],
        "AI_Basics": [88],
        "Attendance": [85.0]
    })
    with pytest.raises(ValueError, match="marks outside"):
        validate_student_data(df, raise_on_error=True)


def test_validation_invalid_marks_non_numeric():
    """Validation must fail if mark is a string/non-numeric."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John"],
        "Department": ["CS"],
        "Python": ["AB"],  # Non-numeric
        "Data_Structures": [80],
        "DBMS": [75],
        "Statistics": [70],
        "AI_Basics": [88],
        "Attendance": [85.0]
    })
    with pytest.raises(ValueError, match="non-numeric"):
        validate_student_data(df, raise_on_error=True)


# ============================================================================
# 7. ATTENDANCE ANALYSIS & THRESHOLD
# ============================================================================

def test_attendance_analysis(sample_valid_df):
    """Verify attendance calculations and low attendance identification (< 75%)."""
    metrics_df = calculate_student_metrics(sample_valid_df)
    graded_df = assign_grades_and_status(metrics_df)

    att_summary = analyze_attendance(graded_df, threshold=75.0)

    assert att_summary["low_attendance_count"] == 2
    assert att_summary["highest_attendance"] == 92.0
    assert att_summary["lowest_attendance"] == 68.0
    assert att_summary["class_avg_attendance"] == 79.8
    assert att_summary["compliant_percentage"] == 60.0


# ============================================================================
# 8. SPECIFIC EDGE CASE SUITE (EMPTY DATA, MISSING CSV, ZERO PASSERS, ETC.)
# ============================================================================

def test_edge_case_empty_dataset_validation():
    """Empty dataframe must be rejected by data validation."""
    empty_df = pd.DataFrame(columns=[
        "Student_ID", "Name", "Department", "Python", "Data_Structures",
        "DBMS", "Statistics", "AI_Basics", "Attendance"
    ])
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_student_data(empty_df, raise_on_error=True)


def test_edge_case_empty_dataset_class_statistics():
    """Calculating class statistics on an empty dataframe must raise ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="no student records"):
        calculate_class_statistics(empty_df)


def test_edge_case_missing_student_names():
    """Blank or whitespace-only student names must be flagged."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["   "],  # Whitespace-only name
        "Department": ["CS"],
        "Python": [80],
        "Data_Structures": [80],
        "DBMS": [80],
        "Statistics": [80],
        "AI_Basics": [80],
        "Attendance": [85.0]
    })
    with pytest.raises(ValueError, match="blank or missing Student Names"):
        validate_student_data(df, raise_on_error=True)


def test_edge_case_missing_student_ids():
    """Blank or whitespace-only student IDs must be flagged."""
    df = pd.DataFrame({
        "Student_ID": [""],  # Empty ID
        "Name": ["John Doe"],
        "Department": ["CS"],
        "Python": [80],
        "Data_Structures": [80],
        "DBMS": [80],
        "Statistics": [80],
        "AI_Basics": [80],
        "Attendance": [85.0]
    })
    with pytest.raises(ValueError, match="blank or missing Student IDs"):
        validate_student_data(df, raise_on_error=True)


def test_edge_case_invalid_attendance_negative():
    """Attendance below 0% must be flagged."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John Doe"],
        "Department": ["CS"],
        "Python": [80],
        "Data_Structures": [80],
        "DBMS": [80],
        "Statistics": [80],
        "AI_Basics": [80],
        "Attendance": [-5.0]  # Negative attendance
    })
    with pytest.raises(ValueError, match="Attendance percentages outside"):
        validate_student_data(df, raise_on_error=True)


def test_edge_case_invalid_attendance_above_100():
    """Attendance above 100% must be flagged."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John Doe"],
        "Department": ["CS"],
        "Python": [80],
        "Data_Structures": [80],
        "DBMS": [80],
        "Statistics": [80],
        "AI_Basics": [80],
        "Attendance": [110.0]  # Above 100%
    })
    with pytest.raises(ValueError, match="Attendance percentages outside"):
        validate_student_data(df, raise_on_error=True)


def test_edge_case_invalid_attendance_non_numeric():
    """Attendance containing non-numeric values must be flagged."""
    df = pd.DataFrame({
        "Student_ID": ["S01"],
        "Name": ["John Doe"],
        "Department": ["CS"],
        "Python": [80],
        "Data_Structures": [80],
        "DBMS": [80],
        "Statistics": [80],
        "AI_Basics": [80],
        "Attendance": ["Ninety"]  # Non-numeric
    })
    with pytest.raises(ValueError, match="non-numeric"):
        validate_student_data(df, raise_on_error=True)


def test_edge_case_no_passing_students():
    """Cohort where every student fails must report 0% pass rate without crashes."""
    failing_df = pd.DataFrame({
        "Student_ID": ["F01", "F02"],
        "Name": ["Fail One", "Fail Two"],
        "Department": ["CS", "AI"],
        "Python": [20, 30],
        "Data_Structures": [25, 28],
        "DBMS": [30, 32],
        "Statistics": [15, 20],
        "AI_Basics": [22, 25],
        "Attendance": [70.0, 60.0]
    })
    metrics_df = calculate_student_metrics(failing_df)
    graded_df = assign_grades_and_status(metrics_df)
    stats = calculate_class_statistics(graded_df)

    assert stats["passed_students"] == 0
    assert stats["failed_students"] == 2
    assert stats["pass_percentage"] == 0.0
    assert all(graded_df["Status"] == "Fail")
    assert all(graded_df["Grade"] == "F")


def test_edge_case_missing_csv_file(tmp_path):
    """Attempting to load a non-existent file must raise FileNotFoundError."""
    from functions import load_student_dataset
    fake_path = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_student_dataset(fake_path)


def test_edge_case_unreadable_or_empty_csv_file(tmp_path):
    """Attempting to load an empty CSV file must raise ValueError."""
    from functions import load_student_dataset
    empty_file = tmp_path / "empty_records.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_student_dataset(str(empty_file))


def test_vercel_entrypoints_and_html_dashboard():
    """Verify that Vercel top-level app and handler generate valid HTTP responses."""
    import main
    # Verify top-level exports required by Vercel
    assert hasattr(main, "app"), "Vercel requires top-level 'app' callable"
    assert hasattr(main, "handler"), "Vercel requires top-level 'handler' class"
    assert hasattr(main, "application"), "WSGI requires 'application' alias"

    # Test WSGI invocation
    captured = {}
    def fake_start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    response = main.app({"REQUEST_METHOD": "GET"}, fake_start_response)
    assert captured["status"] == "200 OK"
    assert any("text/html" in val for key, val in captured["headers"] if key == "Content-Type")
    assert len(response[0]) > 0
    assert b"Student Performance Analytics System" in response[0]


