# Business Insights Report

## Executive Summary

This EDA project analyzes 86,638 cleaned hotel
booking records from a real public dataset. The business focus is demand patterns,
cancellation risk, source-market behavior, customer segmentation, room demand, and an
estimated room-revenue proxy calculated as `ADR x total_nights` for non-canceled bookings.

Key headline findings:

- Overall cancellation rate: **27.7%**
- Highest-volume hotel type: **City Hotel**
- Highest-risk hotel type: **City Hotel**
- Strongest demand month: **August**
- Largest market segment: **Online TA**
- Strongest source country: **PRT**
- Most common room type: **A**

## Business Questions Answered

1. **Which hotel type receives the most bookings?**

   City Hotel leads with 53,043 clean bookings.
2. **What is the overall cancellation rate?**

   The cleaned dataset has an overall cancellation rate of 27.7%.
3. **Which hotel type has the highest cancellation risk?**

   City Hotel has the higher cancellation rate at 30.2%.
4. **Which hotel type contributes the most realized revenue?**

   City Hotel contributes the most estimated realized room revenue: 12,112,961 ADR-units.
5. **Which month has the highest booking demand?**

   August has the highest booking volume with 11,194 bookings.
6. **Which month has the highest cancellation rate?**

   August has the highest cancellation rate at 32.4%.
7. **Which year has the highest booking volume?**

   2016 has the most records with 41,967 bookings.
8. **Which market segment is the main acquisition source?**

   Online TA is the largest segment with 51,285 bookings.
9. **Which distribution channel brings the most bookings?**

   TA/TO is the leading distribution channel with 68,651 bookings.
10. **Which customer type dominates demand?**

   Transient customers dominate with 71,366 bookings.
11. **Which customer type contributes the most realized revenue?**

   Transient contributes the highest estimated realized revenue: 18,322,571 ADR-units.
12. **Which country is the strongest source market?**

   PRT is the top country with 26,864 bookings.
13. **Among top source countries, where is cancellation risk highest?**

   BRA has the highest cancellation rate among the top source countries at 36.5%.
14. **Do longer lead times relate to cancellation behavior?**

   Canceled bookings have a median lead time of 80.0 days versus 38.0 days for non-canceled bookings.
15. **Are repeated guests less likely to cancel?**

   Repeated guests cancel at 8.2%, while first-time guests cancel at 28.4%.
16. **Do special requests indicate stronger booking intent?**

   Bookings with at least one special request cancel at 21.8% compared with 33.5% for bookings with no request.
17. **Which reserved room type is most common?**

   Room type A is most requested with 56,009 bookings.
18. **How often do room assignments change, and does it matter?**

   Room type changes occur in 14.6% of bookings. Changed-room bookings cancel at 4.8% versus 31.6% when the room type stays the same.
19. **Which season is busiest?**

   Summer is the busiest season with 28,903 bookings.
20. **Which season contributes the most estimated realized revenue?**

   Summer contributes the highest estimated realized revenue: 10,515,668 ADR-units.
21. **What length of stay is most common?**

   1-2 nights is the most common stay length category with 32,903 bookings.
22. **Do families cancel differently from non-family bookings?**

   Family bookings cancel at 34.4%, while non-family bookings cancel at 26.9%.
23. **Which deposit type has the highest cancellation rate?**

   Non Refund bookings show the highest cancellation rate at 94.7%.

## Visualization-by-Visualization Insights

### 1. Missing Values Before Cleaning

- Visualization type: Bar Chart
- Screenshot: `Images/01_missing_values_bar.png`
- Insight: Company and agent identifiers account for most missing values, so they are treated as explicit 'No Company' and 'No Agent' categories instead of being dropped. Country and children have small missing counts and are imputed safely.

### 2. Dataset Size Before and After Cleaning

- Visualization type: Bar Chart
- Screenshot: `Images/02_duplicate_records_bar.png`
- Insight: The project starts with 119,390 rows and removes 31,994 exact duplicates, leaving 86,638 clean rows.

### 3. Booking Cancellation Distribution

- Visualization type: Count Plot
- Screenshot: `Images/03_cancellation_distribution.png`
- Insight: The overall cancellation rate is 27.7%, so cancellation is a major operational and revenue-risk theme in the dataset.

### 4. Booking Volume by Hotel Type

- Visualization type: Count Plot
- Screenshot: `Images/04_hotel_type_distribution.png`
- Insight: City Hotel has the larger booking base with 53,043 records, making it the primary volume driver.

### 5. Cancellation Rate by Hotel Type

- Visualization type: Bar Chart
- Screenshot: `Images/05_cancellation_rate_by_hotel.png`
- Insight: City Hotel has the higher cancellation rate at 30.2%, which suggests different risk management strategies by hotel type.

### 6. Monthly Booking Trend

- Visualization type: Line Chart
- Screenshot: `Images/06_monthly_bookings_trend.png`
- Insight: The monthly trend shows demand moving unevenly through time, with visible peaks that can support staffing, pricing, and campaign timing decisions.

### 7. Cancellation Rate by Arrival Month

- Visualization type: Line Chart
- Screenshot: `Images/07_monthly_cancellation_rate.png`
- Insight: August has the highest cancellation rate at 32.4%, making it a priority month for deposit, reminder, or overbooking policies.

### 8. Booking Volume by Calendar Month

- Visualization type: Count Plot
- Screenshot: `Images/08_booking_volume_by_month.png`
- Insight: August is the strongest demand month with 11,194 bookings.

### 9. Bookings by Arrival Year

- Visualization type: Bar Chart
- Screenshot: `Images/09_bookings_by_year.png`
- Insight: 2016 has the highest observed booking count. Because the dataset does not cover every month equally across all years, year comparisons should be interpreted as dataset-period trends.

### 10. Bookings by Market Segment

- Visualization type: Bar Chart
- Screenshot: `Images/10_market_segment_bookings.png`
- Insight: Online TA is the leading acquisition segment with 51,285 bookings.

### 11. Bookings by Distribution Channel

- Visualization type: Bar Chart
- Screenshot: `Images/11_distribution_channel_bookings.png`
- Insight: TA/TO is the main distribution channel with 68,651 bookings.

### 12. Customer Type Share

- Visualization type: Pie Chart
- Screenshot: `Images/12_customer_type_pie.png`
- Insight: Transient customers represent the biggest share of demand, so retention and cancellation controls should be tailored to this group.

### 13. Top 10 Source Countries

- Visualization type: Bar Chart
- Screenshot: `Images/13_top_countries.png`
- Insight: PRT is the strongest source market with 26,864 bookings.

### 14. ADR Distribution After IQR Capping

- Visualization type: Histogram and Distribution Plot
- Screenshot: `Images/14_adr_histogram.png`
- Insight: ADR is right-skewed, so the project keeps the original values but uses IQR-capped ADR for readable visuals and robust revenue comparisons.

### 15. Lead Time Distribution After IQR Capping

- Visualization type: Histogram and Distribution Plot
- Screenshot: `Images/15_lead_time_distribution.png`
- Insight: Lead time is also right-skewed, showing that most bookings are made within a moderate window while a smaller group books far in advance.

### 16. ADR Box Plot by Hotel Type

- Visualization type: Box Plot
- Screenshot: `Images/16_adr_boxplot_by_hotel.png`
- Insight: City Hotel has the higher median capped ADR at 105.7, indicating stronger pricing power.

### 17. Lead Time Box Plot by Cancellation Status

- Visualization type: Box Plot
- Screenshot: `Images/17_lead_time_boxplot_by_status.png`
- Insight: Canceled bookings tend to have longer lead times, which makes lead time a useful early warning feature for cancellation-risk monitoring.

### 18. ADR Violin Plot by Customer Type

- Visualization type: Violin Plot
- Screenshot: `Images/18_adr_violin_by_customer_type.png`
- Insight: The violin plot shows how ADR distributions differ by customer type, helping separate high-volume customer groups from high-rate customer groups.

### 19. Distribution Plot of Length of Stay

- Visualization type: Distribution Plot
- Screenshot: `Images/19_total_nights_distribution.png`
- Insight: 1-2 nights is the most common stay-length category, which is useful for room inventory planning and housekeeping schedules.

### 20. Correlation Matrix Heatmap

- Visualization type: Heatmap and Correlation Matrix
- Screenshot: `Images/20_correlation_heatmap.png`
- Insight: The correlation matrix highlights directional relationships: special requests move opposite to cancellation, while lead time is positively associated with risk.

### 21. Pair Plot of Key Numeric Features

- Visualization type: Pair Plot
- Screenshot: `Images/21_pair_plot_numeric_features.png`
- Insight: The pair plot shows feature interactions and confirms that canceled and non-canceled bookings overlap heavily, so business rules should combine several indicators.

### 22. Lead Time vs ADR by Cancellation Status

- Visualization type: Scatter Plot
- Screenshot: `Images/22_lead_time_vs_adr_scatter.png`
- Insight: The scatter plot shows cancellation status across pricing and lead-time space, surfacing long-lead bookings as a visible risk cluster.

### 23. Estimated Realized Room Revenue by Hotel Type

- Visualization type: Bar Chart
- Screenshot: `Images/23_realized_revenue_by_hotel.png`
- Insight: City Hotel produces the largest estimated realized room revenue at 12,112,961 ADR-units.

### 24. Top Reserved Room Types

- Visualization type: Bar Chart
- Screenshot: `Images/24_room_type_demand.png`
- Insight: Reserved room type A is the dominant room product, so availability and pricing decisions around this type carry high impact.

### 25. Cancellation Rate by Number of Special Requests

- Visualization type: Bar Chart
- Screenshot: `Images/25_special_requests_vs_cancellation.png`
- Insight: Cancellation rate declines as special requests increase, suggesting requests are a practical signal of guest commitment.

### 26. Seasonal Booking Patterns by Hotel Type

- Visualization type: Grouped Bar Chart
- Screenshot: `Images/26_seasonal_booking_patterns.png`
- Insight: Summer is the busiest season, while Summer contributes the most estimated realized revenue.


## Outlier Detection Summary

The project uses IQR outlier flags for numeric fields that are important for hotel
operations and revenue analysis. Outliers are not blindly removed because extreme lead
times, long stays, and high ADR bookings can be real business events. For readable
visuals, ADR and lead time are capped only in visualization-specific fields.

| Column | Q1 | Q3 | Lower Bound | Upper Bound | Outlier Count | Outlier Rate |
|---|---:|---:|---:|---:|---:|---:|
| lead_time | 12.00 | 126.00 | -159.00 | 297.00 | 2,353 | 2.7% |
| adr | 72.90 | 134.44 | -19.41 | 226.74 | 2,511 | 2.9% |
| total_nights | 2.00 | 5.00 | -2.50 | 9.50 | 2,990 | 3.5% |
| total_guests | 2.00 | 2.00 | 2.00 | 2.00 | 29,939 | 34.6% |
| days_in_waiting_list | 0.00 | 0.00 | 0.00 | 0.00 | 854 | 1.0% |
| previous_cancellations | 0.00 | 0.00 | 0.00 | 0.00 | 1,677 | 1.9% |

## Recommended Actions

1. Use lead time, deposit type, special requests, and customer type as early cancellation
   risk signals.
2. Build segment-specific retention playbooks for the dominant market segment and
   customer type.
3. Protect high-demand months with tighter inventory controls and cancellation policies.
4. Monitor source countries with high cancellation risk among top-volume markets.
5. Use room type demand and room assignment changes to improve inventory allocation.
