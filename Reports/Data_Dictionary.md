# Data Dictionary

Dataset: Hotel Booking Demand

Source CSV: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv

## Original Columns

| Column | Description | Cleaned dtype |
|---|---|---|
| `hotel` | Hotel type: City Hotel or Resort Hotel. | `category` |
| `is_canceled` | Cancellation flag. 1 means canceled, 0 means not canceled. | `int64` |
| `lead_time` | Days between booking date and arrival date. | `int64` |
| `arrival_date_year` | Arrival year. | `int64` |
| `arrival_date_month` | Arrival month name. | `category` |
| `arrival_date_week_number` | Arrival ISO week number. | `int64` |
| `arrival_date_day_of_month` | Arrival day of month. | `int64` |
| `stays_in_weekend_nights` | Number of weekend nights booked. | `int64` |
| `stays_in_week_nights` | Number of week nights booked. | `int64` |
| `adults` | Number of adults. | `int64` |
| `children` | Number of children. | `int64` |
| `babies` | Number of babies. | `int64` |
| `meal` | Meal package type. | `category` |
| `country` | Guest country of origin. | `category` |
| `market_segment` | Market segment designation. | `category` |
| `distribution_channel` | Booking distribution channel. | `category` |
| `is_repeated_guest` | Repeated guest flag. | `int64` |
| `previous_cancellations` | Previous canceled bookings by the guest. | `int64` |
| `previous_bookings_not_canceled` | Previous non-canceled bookings by the guest. | `int64` |
| `reserved_room_type` | Reserved room category code. | `category` |
| `assigned_room_type` | Assigned room category code. | `category` |
| `booking_changes` | Number of changes made to the booking. | `int64` |
| `deposit_type` | Deposit policy type. | `category` |
| `agent` | Travel agency identifier. | `category` |
| `company` | Company/entity identifier. | `category` |
| `days_in_waiting_list` | Days the booking stayed on the waiting list. | `int64` |
| `customer_type` | Customer contract/category type. | `category` |
| `adr` | Average Daily Rate, calculated per occupied room night. | `float64` |
| `required_car_parking_spaces` | Number of required parking spaces. | `int64` |
| `total_of_special_requests` | Number of special requests made by the guest. | `int64` |
| `reservation_status` | Final reservation status. | `category` |
| `reservation_status_date` | Date of final reservation status. | `datetime64[ns]` |

## Engineered Columns

| Column | Description | Cleaned dtype |
|---|---|---|
| `arrival_date` | Parsed arrival date. | `datetime64[ns]` |
| `arrival_month_number` | Numeric arrival month. | `int64` |
| `total_nights` | Weekend nights plus week nights. | `int64` |
| `total_guests` | Adults plus children plus babies. | `int64` |
| `is_canceled_label` | Human-readable cancellation status. | `category` |
| `is_repeated_guest_label` | Human-readable repeated guest status. | `category` |
| `has_children` | Binary flag for bookings with children or babies. | `int64` |
| `has_children_label` | Human-readable child presence label. | `category` |
| `is_family` | Family/non-family booking flag. | `category` |
| `room_type_changed` | Flag for reserved room type differing from assigned type. | `int64` |
| `room_type_changed_label` | Human-readable room assignment change label. | `category` |
| `booking_season` | Season inferred from arrival month. | `category` |
| `arrival_quarter` | Quarter inferred from arrival date. | `category` |
| `arrival_year_month` | Year-month period used for trend charts. | `object` |
| `weekend_share` | Share of stay nights that fall on weekends. | `float64` |
| `estimated_revenue` | ADR multiplied by total nights. | `float64` |
| `realized_revenue` | Estimated revenue retained only for non-canceled bookings. | `float64` |
| `length_of_stay_category` | Binned length-of-stay segment. | `category` |
| `lead_time_segment` | Binned lead-time segment. | `category` |
| `adr_segment` | Binned ADR segment. | `category` |
| `adr_capped` | ADR capped at IQR bounds for robust visualization. | `float64` |
| `lead_time_capped` | Lead time capped at IQR bounds for robust visualization. | `int64` |
| `estimated_revenue_capped` | IQR-capped ADR multiplied by total nights. | `float64` |
| `realized_revenue_capped` | Capped revenue proxy retained only for non-canceled bookings. | `float64` |
| `any_iqr_outlier` | Flag for any selected numeric IQR outlier. | `int64` |

## IQR Outlier Summary

Outliers are flagged using the standard IQR rule:

`lower_bound = Q1 - 1.5 * IQR`

`upper_bound = Q3 + 1.5 * IQR`

| Column | Q1 | Q3 | Lower Bound | Upper Bound | Outlier Count | Outlier Rate |
|---|---:|---:|---:|---:|---:|---:|
| lead_time | 12.00 | 126.00 | -159.00 | 297.00 | 2,353 | 2.7% |
| adr | 72.90 | 134.44 | -19.41 | 226.74 | 2,511 | 2.9% |
| total_nights | 2.00 | 5.00 | -2.50 | 9.50 | 2,990 | 3.5% |
| total_guests | 2.00 | 2.00 | 2.00 | 2.00 | 29,939 | 34.6% |
| days_in_waiting_list | 0.00 | 0.00 | 0.00 | 0.00 | 854 | 1.0% |
| previous_cancellations | 0.00 | 0.00 | 0.00 | 0.00 | 1,677 | 1.9% |
