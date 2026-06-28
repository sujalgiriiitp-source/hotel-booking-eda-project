# Hotel Booking Demand - End-to-End Exploratory Data Analysis

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-4C72B0?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)

## Project Overview

This is a complete end-to-end Exploratory Data Analysis project built on the real
**Hotel Booking Demand** public dataset. The project is designed as a professional
Data Analyst internship portfolio project and covers data cleaning, feature engineering,
outlier detection, visual analysis, business-question answering, and executive reporting.

The dataset contains booking records for a city hotel and a resort hotel. It is widely
used for hotel demand, cancellation, revenue, and customer behavior analysis.

## Dataset Source

- Dataset mirror used in this project: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv
- TidyTuesday source page: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11
- Original Kaggle dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Original paper DOI: https://doi.org/10.1016/j.dib.2018.11.126

## Dataset Size

| Metric | Value |
|---|---:|
| Raw rows | 119,390 |
| Raw columns | 32 |
| Exact duplicates removed | 31,994 |
| Clean rows | 86,638 |
| Clean columns | 64 |
| Minimum record requirement | 20,000+ |
| Requirement met | Yes |

## Project Structure

```text
EDA_Project/
|
|-- Dataset/
|   |-- hotel_bookings.csv
|   |-- hotel_bookings_cleaned.csv
|-- Notebook/
|   |-- Hotel_Booking_Demand_EDA.ipynb
|-- Images/
|   |-- 26 chart screenshots
|-- Reports/
|   |-- Data_Dictionary.md
|   |-- Business_Insights_Report.md
|   |-- Final_Conclusion.md
|   |-- analysis_summary.json
|-- scripts/
|   |-- hotel_booking_eda.py
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- .gitignore
```

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Cleaning and Feature Engineering

The project performs:

- Missing-value treatment for `company`, `agent`, `country`, and `children`
- Exact duplicate removal
- Date parsing for arrival and reservation-status dates
- Data type correction for numeric, date, and categorical variables
- Invalid row handling for zero guests, zero-night bookings, negative ADR, and invalid dates
- IQR outlier detection for key numeric fields
- Visualization-safe capping for ADR and lead time
- Feature engineering for:
  - `total_nights`
  - `total_guests`
  - `arrival_date`
  - `booking_season`
  - `arrival_quarter`
  - `arrival_year_month`
  - `is_family`
  - `room_type_changed`
  - `lead_time_segment`
  - `length_of_stay_category`
  - `estimated_revenue`
  - `realized_revenue`

## Key Business Findings

- Overall cancellation rate: **27.7%**
- Highest-volume hotel type: **City Hotel**
- Highest cancellation-risk hotel type: **City Hotel**
- Strongest booking month: **August**
- Leading market segment: **Online TA**
- Leading distribution channel: **TA/TO**
- Strongest source country: **PRT**
- Most requested room type: **A**
- Highest realized revenue hotel type: **City Hotel**

## Business Questions Answered

| # | Business Question | Answer |
|---:|---|---|
| 1 | Which hotel type receives the most bookings? | City Hotel leads with 53,043 clean bookings. |
| 2 | What is the overall cancellation rate? | The cleaned dataset has an overall cancellation rate of 27.7%. |
| 3 | Which hotel type has the highest cancellation risk? | City Hotel has the higher cancellation rate at 30.2%. |
| 4 | Which hotel type contributes the most realized revenue? | City Hotel contributes the most estimated realized room revenue: 12,112,961 ADR-units. |
| 5 | Which month has the highest booking demand? | August has the highest booking volume with 11,194 bookings. |
| 6 | Which month has the highest cancellation rate? | August has the highest cancellation rate at 32.4%. |
| 7 | Which year has the highest booking volume? | 2016 has the most records with 41,967 bookings. |
| 8 | Which market segment is the main acquisition source? | Online TA is the largest segment with 51,285 bookings. |
| 9 | Which distribution channel brings the most bookings? | TA/TO is the leading distribution channel with 68,651 bookings. |
| 10 | Which customer type dominates demand? | Transient customers dominate with 71,366 bookings. |
| 11 | Which customer type contributes the most realized revenue? | Transient contributes the highest estimated realized revenue: 18,322,571 ADR-units. |
| 12 | Which country is the strongest source market? | PRT is the top country with 26,864 bookings. |
| 13 | Among top source countries, where is cancellation risk highest? | BRA has the highest cancellation rate among the top source countries at 36.5%. |
| 14 | Do longer lead times relate to cancellation behavior? | Canceled bookings have a median lead time of 80.0 days versus 38.0 days for non-canceled bookings. |
| 15 | Are repeated guests less likely to cancel? | Repeated guests cancel at 8.2%, while first-time guests cancel at 28.4%. |
| 16 | Do special requests indicate stronger booking intent? | Bookings with at least one special request cancel at 21.8% compared with 33.5% for bookings with no request. |
| 17 | Which reserved room type is most common? | Room type A is most requested with 56,009 bookings. |
| 18 | How often do room assignments change, and does it matter? | Room type changes occur in 14.6% of bookings. Changed-room bookings cancel at 4.8% versus 31.6% when the room type stays the same. |
| 19 | Which season is busiest? | Summer is the busiest season with 28,903 bookings. |
| 20 | Which season contributes the most estimated realized revenue? | Summer contributes the highest estimated realized revenue: 10,515,668 ADR-units. |
| 21 | What length of stay is most common? | 1-2 nights is the most common stay length category with 32,903 bookings. |
| 22 | Do families cancel differently from non-family bookings? | Family bookings cancel at 34.4%, while non-family bookings cancel at 26.9%. |
| 23 | Which deposit type has the highest cancellation rate? | Non Refund bookings show the highest cancellation rate at 94.7%. |

## Visualization Gallery and Insights

### 1. Missing Values Before Cleaning

**Type:** Bar Chart

![Missing Values Before Cleaning](Images/01_missing_values_bar.png)

**Insight:** Company and agent identifiers account for most missing values, so they are treated as explicit 'No Company' and 'No Agent' categories instead of being dropped. Country and children have small missing counts and are imputed safely.

### 2. Dataset Size Before and After Cleaning

**Type:** Bar Chart

![Dataset Size Before and After Cleaning](Images/02_duplicate_records_bar.png)

**Insight:** The project starts with 119,390 rows and removes 31,994 exact duplicates, leaving 86,638 clean rows.

### 3. Booking Cancellation Distribution

**Type:** Count Plot

![Booking Cancellation Distribution](Images/03_cancellation_distribution.png)

**Insight:** The overall cancellation rate is 27.7%, so cancellation is a major operational and revenue-risk theme in the dataset.

### 4. Booking Volume by Hotel Type

**Type:** Count Plot

![Booking Volume by Hotel Type](Images/04_hotel_type_distribution.png)

**Insight:** City Hotel has the larger booking base with 53,043 records, making it the primary volume driver.

### 5. Cancellation Rate by Hotel Type

**Type:** Bar Chart

![Cancellation Rate by Hotel Type](Images/05_cancellation_rate_by_hotel.png)

**Insight:** City Hotel has the higher cancellation rate at 30.2%, which suggests different risk management strategies by hotel type.

### 6. Monthly Booking Trend

**Type:** Line Chart

![Monthly Booking Trend](Images/06_monthly_bookings_trend.png)

**Insight:** The monthly trend shows demand moving unevenly through time, with visible peaks that can support staffing, pricing, and campaign timing decisions.

### 7. Cancellation Rate by Arrival Month

**Type:** Line Chart

![Cancellation Rate by Arrival Month](Images/07_monthly_cancellation_rate.png)

**Insight:** August has the highest cancellation rate at 32.4%, making it a priority month for deposit, reminder, or overbooking policies.

### 8. Booking Volume by Calendar Month

**Type:** Count Plot

![Booking Volume by Calendar Month](Images/08_booking_volume_by_month.png)

**Insight:** August is the strongest demand month with 11,194 bookings.

### 9. Bookings by Arrival Year

**Type:** Bar Chart

![Bookings by Arrival Year](Images/09_bookings_by_year.png)

**Insight:** 2016 has the highest observed booking count. Because the dataset does not cover every month equally across all years, year comparisons should be interpreted as dataset-period trends.

### 10. Bookings by Market Segment

**Type:** Bar Chart

![Bookings by Market Segment](Images/10_market_segment_bookings.png)

**Insight:** Online TA is the leading acquisition segment with 51,285 bookings.

### 11. Bookings by Distribution Channel

**Type:** Bar Chart

![Bookings by Distribution Channel](Images/11_distribution_channel_bookings.png)

**Insight:** TA/TO is the main distribution channel with 68,651 bookings.

### 12. Customer Type Share

**Type:** Pie Chart

![Customer Type Share](Images/12_customer_type_pie.png)

**Insight:** Transient customers represent the biggest share of demand, so retention and cancellation controls should be tailored to this group.

### 13. Top 10 Source Countries

**Type:** Bar Chart

![Top 10 Source Countries](Images/13_top_countries.png)

**Insight:** PRT is the strongest source market with 26,864 bookings.

### 14. ADR Distribution After IQR Capping

**Type:** Histogram and Distribution Plot

![ADR Distribution After IQR Capping](Images/14_adr_histogram.png)

**Insight:** ADR is right-skewed, so the project keeps the original values but uses IQR-capped ADR for readable visuals and robust revenue comparisons.

### 15. Lead Time Distribution After IQR Capping

**Type:** Histogram and Distribution Plot

![Lead Time Distribution After IQR Capping](Images/15_lead_time_distribution.png)

**Insight:** Lead time is also right-skewed, showing that most bookings are made within a moderate window while a smaller group books far in advance.

### 16. ADR Box Plot by Hotel Type

**Type:** Box Plot

![ADR Box Plot by Hotel Type](Images/16_adr_boxplot_by_hotel.png)

**Insight:** City Hotel has the higher median capped ADR at 105.7, indicating stronger pricing power.

### 17. Lead Time Box Plot by Cancellation Status

**Type:** Box Plot

![Lead Time Box Plot by Cancellation Status](Images/17_lead_time_boxplot_by_status.png)

**Insight:** Canceled bookings tend to have longer lead times, which makes lead time a useful early warning feature for cancellation-risk monitoring.

### 18. ADR Violin Plot by Customer Type

**Type:** Violin Plot

![ADR Violin Plot by Customer Type](Images/18_adr_violin_by_customer_type.png)

**Insight:** The violin plot shows how ADR distributions differ by customer type, helping separate high-volume customer groups from high-rate customer groups.

### 19. Distribution Plot of Length of Stay

**Type:** Distribution Plot

![Distribution Plot of Length of Stay](Images/19_total_nights_distribution.png)

**Insight:** 1-2 nights is the most common stay-length category, which is useful for room inventory planning and housekeeping schedules.

### 20. Correlation Matrix Heatmap

**Type:** Heatmap and Correlation Matrix

![Correlation Matrix Heatmap](Images/20_correlation_heatmap.png)

**Insight:** The correlation matrix highlights directional relationships: special requests move opposite to cancellation, while lead time is positively associated with risk.

### 21. Pair Plot of Key Numeric Features

**Type:** Pair Plot

![Pair Plot of Key Numeric Features](Images/21_pair_plot_numeric_features.png)

**Insight:** The pair plot shows feature interactions and confirms that canceled and non-canceled bookings overlap heavily, so business rules should combine several indicators.

### 22. Lead Time vs ADR by Cancellation Status

**Type:** Scatter Plot

![Lead Time vs ADR by Cancellation Status](Images/22_lead_time_vs_adr_scatter.png)

**Insight:** The scatter plot shows cancellation status across pricing and lead-time space, surfacing long-lead bookings as a visible risk cluster.

### 23. Estimated Realized Room Revenue by Hotel Type

**Type:** Bar Chart

![Estimated Realized Room Revenue by Hotel Type](Images/23_realized_revenue_by_hotel.png)

**Insight:** City Hotel produces the largest estimated realized room revenue at 12,112,961 ADR-units.

### 24. Top Reserved Room Types

**Type:** Bar Chart

![Top Reserved Room Types](Images/24_room_type_demand.png)

**Insight:** Reserved room type A is the dominant room product, so availability and pricing decisions around this type carry high impact.

### 25. Cancellation Rate by Number of Special Requests

**Type:** Bar Chart

![Cancellation Rate by Number of Special Requests](Images/25_special_requests_vs_cancellation.png)

**Insight:** Cancellation rate declines as special requests increase, suggesting requests are a practical signal of guest commitment.

### 26. Seasonal Booking Patterns by Hotel Type

**Type:** Grouped Bar Chart

![Seasonal Booking Patterns by Hotel Type](Images/26_seasonal_booking_patterns.png)

**Insight:** Summer is the busiest season, while Summer contributes the most estimated realized revenue.

## How to Run This Project

1. Clone or download this repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open and run:

```bash
jupyter notebook Notebook/Hotel_Booking_Demand_EDA.ipynb
```

To regenerate the complete project assets from the script:

```bash
python scripts/hotel_booking_eda.py
```

## Reports

- [Data Dictionary](Reports/Data_Dictionary.md)
- [Business Insights Report](Reports/Business_Insights_Report.md)
- [Final Conclusion](Reports/Final_Conclusion.md)

## Portfolio Value

This project demonstrates real-world analyst skills:

- Working with a public dataset above 20,000 records
- Cleaning messy hospitality data
- Building reproducible feature engineering
- Detecting and handling outliers using IQR
- Producing 20+ professional visualizations
- Translating analysis into business recommendations
- Packaging the project for GitHub and resume use

## 👤 Author

**Sujal Giri**

- GitHub: https://github.com/sujalgiriiitp-source
- LinkedIn: https://www.linkedin.com/in/sujal-giri-9501253a0
- Email: sujalgiriiitp@gmail.com
