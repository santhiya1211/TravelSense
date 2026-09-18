# Job Market Analytics Dashboard

An end-to-end data analytics project exploring ~123,000 real job postings (LinkedIn Job Postings dataset, 2023–2024) to uncover patterns in skill demand, salary, remote work availability, and hiring activity by company and location.

![Dashboard Overview](screenshots/page1_overview.png)

## Project Overview

This project answers questions like:
- Which skills are most in-demand across job postings?
- How does salary scale with seniority?
- Does company size affect pay?
- Which companies and cities post the most jobs?
- How well is remote work actually documented in job listings?

## Tools Used

- **MySQL** — data cleaning, relational schema design, and analysis queries
- **Python (pandas, SQLAlchemy)** — reliable bulk CSV-to-database loading
- **Power BI** — interactive two-page dashboard with live database connection
- **DAX** — calculated columns for cleaning up unlabeled/blank categories

## Data

Source: [LinkedIn Job Postings (2023–2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) — Kaggle

The raw dataset was split across 4 CSVs (`postings`, `companies`, `skills`, `job_skills`) and loaded into a normalized MySQL database with proper relationships:

```
postings.job_id      → job_skills.job_id
job_skills.skill_abr  → skills.skill_abr
postings.company_id   → companies.company_id
```

## Analysis

15+ SQL queries were written to answer core business questions, including:
- Top in-demand skills overall and by job title
- Average salary by experience level
- Remote vs. on-site distribution
- Top hiring companies and locations
- Skill co-occurrence (which skills most often appear together in the same posting)

See [`queries.sql`](queries.sql) for the full query set.

## Dashboard

**Page 1 — Overview**
- KPI cards: total postings, unique companies, average salary, skills tracked
- Top 10 in-demand skills
- Remote work status breakdown
- Salary by experience level
- Interactive job-title filter

**Page 2 — Companies & Locations**
- Top 10 hiring companies
- Top 10 locations by posting volume
- Average salary by company size

## Key Findings

1. **Broad business skills dominate demand** — Information Technology, Sales, and Management top the list, ahead of narrower technical skills.
2. **Remote work status is inconsistently reported** — only 12.3% of postings explicitly marked remote eligibility.
3. **Salary scales predictably with seniority** — a clean downward trend from Executive to Internship level.
4. **Company size doesn't guarantee higher pay** — mid-sized companies (bracket 3) show the highest average salaries in the dataset.
5. **Hiring is concentrated** — a small set of staffing firms and metro areas account for a disproportionate share of postings.

Full write-up: [`Job_Market_Insights_Report.docx`](Job_Market_Insights_Report.docx)

## Data Quality Notes

- `remote_allowed` and `formatted_experience_level` had significant missing data; blanks were explicitly relabeled ("Not Specified") rather than hidden, to avoid misrepresenting the data.
- `company_size` is provided as a numeric bracket (1–7); exact employee-count ranges were not independently confirmed for this analysis.

## Repository Structure

```
├── README.md
├── queries.sql
├── Job_Market_Insights_Report.docx
└── screenshots/
    ├── page1_overview.png
    └── page2_companies_locations.png
```

## Author

Santhiya — [LinkedIn](https://www.linkedin.com/in/santhiya-s-3021562b1/)
