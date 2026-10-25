# Student Performance Analytics System

An end-to-end data analytics and reporting system built with **Python**, **NumPy**, and **Pandas** for the **EWB Courses Python with AI** project assignment.

---

## 📌 Project Overview

The **Student Performance Analytics System** automates the evaluation of student academic performance and attendance compliance. It processes academic records, validates data integrity, performs statistical computations using NumPy, applies project-defined grading and pass/fail rules, determines top performers with dense-ranking tie handling, and generates actionable, data-driven analytical insights.

---

## 🚀 Key Features

1. **Automated CSV Data Ingestion & Validation**:
   - Checks for required columns.
   - Detects missing/null values (`NaN`).
   - Identifies duplicate Student IDs.
   - Validates marks range `[0, 100]` and attendance range `[0, 100]`.
   - Prevents non-numeric marks from corrupting downstream calculations.

2. **NumPy-Powered Numerical Computations**:
   - Vectorized row-wise summation (`np.sum`) and row-wise mean (`np.mean`) across 5 subjects.
   - Fast subject-level aggregation: mean, min, max, standard deviation (`np.std`).
   - Class-wide metric calculation: highest score, lowest score, overall mean.

3. **Transparent Grading & Academic Pass/Fail Model**:
   - **Academic Pass Condition**: A student must score at least **35 marks** in **every** subject.
   - **Grading Scale** (Project-Defined Rules):
     - **A**: Average $\ge 90\%$
     - **B**: Average $\ge 75\%$ and $< 90\%$
     - **C**: Average $\ge 60\%$ and $< 75\%$
     - **D**: Average $\ge 50\%$ and $< 60\%$
     - **E**: Average $\ge 35\%$ and $< 50\%$
     - **F**: Fail if any subject mark is below 35 or average $< 35\%$
   - *Note*: These grading rules are project-defined for this assignment and do not represent official university rules.

4. **Independent Attendance Auditing**:
   - Tracks institutional attendance compliance (minimum threshold: $75\%$).
   - Explicitly decoupled: attendance compliance does not alter academic pass/fail status, but generates an administrative counseling alert for students with low attendance.

5. **Dense-Rank Top Performer Identification**:
   - Accurately handles ties among top performers using dense ranking (`rank(method='dense')`), ensuring all students who tie for top positions are included without arbitrary truncation.

6. **Comprehensive Automated Test Suite**:
   - **38 automated unit, integration, and edge-case tests** using `pytest` covering calculations, grade boundaries, edge cases (empty dataset, missing names/IDs, negative/excess attendance, zero passing students, missing CSV files), data validation, and tie handling.

---

## 🛠️ Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.12+** | Core programming language |
| **Pandas** | Tabular data structures, filtering, sorting, and CSV handling |
| **NumPy** | High-performance vectorized numerical operations and statistics |
| **Tabulate** | Clean ASCII table formatting for presentation-grade CLI displays |
| **PyTest** | Automated unit testing, edge case validation, and boundary verification |

---

## 📁 Project Structure

```
EWB PythonAI/
│
├── main.py                   # Main orchestration script and CLI report generator
├── functions.py              # Modular, reusable analytical & validation functions
├── students.csv              # Fictional dataset containing 22 student records
├── test_analytics.py         # PyTest test suite (38 automated tests)
├── requirements.txt          # Python dependencies
├── README.md                 # Project README and quickstart guide
├── Project_Documentation.md  # In-depth academic report & trainer presentation guide
└── .gitignore                # Git ignore rules for Python artifacts
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- `pip` package manager.

### 1. Clone or Open Workspace
Navigate to the project root directory:
```bash
cd "c:\Users\hi\Desktop\EWB PythonAI"
```

### 2. (Optional) Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Project

Execute the main application with a single command:
```bash
python main.py
```

### Expected Output Sections:
1. **Dataset Overview**: Displays the first 5 fictional student records and dataset dimensions (22 records, 9 attributes).
2. **Student Results**: Complete table showing Total Marks (out of 500), Average Marks (%), Grade (A–F), Status (Pass/Fail), and Attendance.
3. **Class Performance Statistics**: High/low/average totals, named highest and lowest performers, pass percentage, and grade distribution.
4. **Subject-Wise Analysis**: Mean, highest, lowest, standard deviation, pass rate, and explicit Best-Performing & Most Challenging subjects.
5. **Top Performers**: Top 3 ranks with tie-handling.
6. **Attendance Summary**: Institutional compliance rate and list of students flagged for low attendance ($< 75\%$).
7. **Key Analytical Observations**: Concrete, data-driven takeaways.

---

## 🧪 Running the Automated Tests

Run the full automated test suite using `pytest`:

```bash
pytest test_analytics.py -v
```

To run quietly:
```bash
pytest test_analytics.py -q
```

All **39 tests** pass, validating:
- Arithmetic sum and mean against manual calculations.
- Exact boundary conditions for Grades A, B, C, D, E, and F.
- Pass/Fail evaluation when a student scores below 35 in a single subject despite high averages.
- Subject-level aggregations, pass rates, and best/worst subject detection.
- Multi-way tie handling in top performer rankings using dense ranking.
- Detection of missing columns, null cells, blank names, blank student IDs, duplicate Student IDs, negative marks, marks $> 100$, and non-numeric inputs.
- Edge cases: Empty datasets, cohorts where zero students pass, missing CSV files, and unreadable/empty CSV files.
- Attendance compliance calculations and negative/excess attendance filtering.
- Vercel Serverless Function entrypoints (`app`, `handler`, and `generate_html_dashboard`).

---

## ☁️ Vercel Cloud Deployment

The application is configured to run both as a local CLI tool and as a serverless web application on **Vercel**:
- **Entrypoints**: Top-level `app` (WSGI) and `handler` (`BaseHTTPRequestHandler`) in `main.py`.
- **Web Dashboard**: Visiting the deployed Vercel URL displays a responsive, modern HTML dashboard presenting all analytics tables, metrics cards, and observations.
- **Zero Configuration**: Connect your GitHub repository to Vercel and it will automatically build and deploy.

---

## 📊 Summary of Actual Project Results

From running `main.py` on the 22-record fictional dataset:
- **Total Students**: 22
- **Passed**: 19 ($86.36\%$)
- **Failed**: 3 ($13.64\%$)
- **Highest Performer**: Aarav Sharma (`STD101`) – $475.0 / 500$, $95.0\%$, Grade A
- **Lowest Performer**: Siddharth Rao (`STD117`) – $180.0 / 500$, $36.0\%$, Grade F
- **Top Performing Subject**: DBMS ($70.45$ class average, $100\%$ pass rate)
- **Most Challenging Subject**: Python ($66.41$ class average, $90.91\%$ pass rate)
- **Attendance Compliance**: $77.27\%$ compliant; 5 students below the $75\%$ attendance threshold.
- **Attendance Impact**: Regular attendees ($\ge 75\%$) achieved an average mark of **$71.88\%$**, compared to **$55.72\%$** for low-attendance students (**$+16.16\%$** advantage).

---

## 🐙 GitHub Preparation & Upload Guide

> **Note**: Per assignment instructions, do not push to GitHub without explicit user approval. Below are the steps prepared for when you are ready to upload.

### Step 1: Initialize Git Repository
```bash
git init
```

### Step 2: Stage and Commit All Files
```bash
git add .
git commit -m "feat: complete Student Performance Analytics System for EWB assignment"
```

### Step 3: Link to Remote GitHub Repository
```bash
git branch -M main
git remote add origin https://github.com/srikarbabuvaddi-star/EWB-PythonAI.git
```

### Step 4: Push to GitHub
```bash
git push -u origin main
```

**GitHub Repository URL**:
`https://github.com/srikarbabuvaddi-star/EWB-PythonAI.git`

---

## 👥 Presentation Talking Points for Your Trainer

1. **Architecture & Clean Code**: Modular design separating validation, calculation, analytics, and presentation (`functions.py` vs `main.py`).
2. **Why NumPy**: Vectorized operations (`np.sum`, `np.mean`) across 2D arrays are drastically faster and cleaner than manual nested loops for numerical datasets.
3. **Why Pandas**: Provides expressive DataFrame capabilities for filtering, grouping, sorting, and tabular data management.
4. **Real-world Policy Edge Cases**: Demonstrated that a student with high marks (e.g. Nikhil Gupta, $73.4\%$ average) still receives Grade F because of failing a single subject ($28$ in Statistics), showing strict adherence to institutional pass criteria.
5. **Fair Tie Handling**: Explained why simple list slicing (`head(3)`) drops tied students and why dense ranking is essential in academic honors.
