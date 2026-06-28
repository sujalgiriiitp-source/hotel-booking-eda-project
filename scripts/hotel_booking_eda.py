"""End-to-end EDA project builder for the Hotel Booking Demand dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MPL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Dataset"
NOTEBOOK_DIR = PROJECT_ROOT / "Notebook"
IMAGE_DIR = PROJECT_ROOT / "Images"
REPORT_DIR = PROJECT_ROOT / "Reports"

RAW_DATA_PATH = DATA_DIR / "hotel_bookings.csv"
CLEAN_DATA_PATH = DATA_DIR / "hotel_bookings_cleaned.csv"
SUMMARY_JSON_PATH = REPORT_DIR / "analysis_summary.json"
NOTEBOOK_PATH = NOTEBOOK_DIR / "Hotel_Booking_Demand_EDA.ipynb"

DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_MAP = {month: index for index, month in enumerate(MONTH_ORDER, start=1)}
SEASON_MAP = {
    "December": "Winter",
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette="deep",
    rc={
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.titlesize": 13,
    },
)


def ensure_directories() -> None:
    """Create all project output directories."""
    for directory in [DATA_DIR, NOTEBOOK_DIR, IMAGE_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def percent(value: float) -> str:
    """Format a decimal as a percentage."""
    return f"{value * 100:.1f}%"


def number(value: float) -> str:
    """Format a number with comma separators."""
    return f"{value:,.0f}"


def decimal(value: float, digits: int = 1) -> str:
    """Format a decimal with a fixed number of places."""
    return f"{value:,.{digits}f}"


def revenue(value: float) -> str:
    """Format estimated room revenue proxy values."""
    return f"{value:,.0f} ADR-units"


def title_case(text: str) -> str:
    """Make dataframe labels easier to read in charts."""
    return str(text).replace("_", " ").title()


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the real hotel bookings dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from {DATA_URL}."
        )
    return pd.read_csv(path)


def calculate_iqr_bounds(series: pd.Series) -> dict[str, float]:
    """Calculate IQR outlier bounds for a numeric series."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper}


def clean_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.Series]:
    """Clean raw data, engineer features, and flag IQR outliers."""
    missing_before = raw_df.isna().sum().sort_values(ascending=False)
    duplicates_before = int(raw_df.duplicated().sum())

    df = raw_df.copy()
    df = df.drop_duplicates().copy()

    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0).astype(int).astype(str)
    df["company"] = df["company"].fillna(0).astype(int).astype(str)
    df["agent"] = df["agent"].replace({"0": "No Agent"})
    df["company"] = df["company"].replace({"0": "No Company"})
    df["meal"] = df["meal"].replace({"Undefined": "SC"})
    df["market_segment"] = df["market_segment"].replace({"Undefined": "Unknown"})
    df["distribution_channel"] = df["distribution_channel"].replace(
        {"Undefined": "Unknown"}
    )

    df["arrival_month_number"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_date"] = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": df["arrival_month_number"],
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"],
        errors="coerce",
    )

    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    invalid_guest_rows = int((df["total_guests"] <= 0).sum())
    invalid_night_rows = int((df["total_nights"] <= 0).sum())
    negative_adr_rows = int((df["adr"] < 0).sum())
    invalid_date_rows = int(df["arrival_date"].isna().sum())
    invalid_mask = (
        (df["total_guests"] <= 0)
        | (df["total_nights"] <= 0)
        | (df["adr"] < 0)
        | df["arrival_date"].isna()
    )
    df = df.loc[~invalid_mask].copy()

    df["is_canceled_label"] = np.where(
        df["is_canceled"].eq(1),
        "Canceled",
        "Not Canceled",
    )
    df["is_repeated_guest_label"] = np.where(
        df["is_repeated_guest"].eq(1),
        "Repeated Guest",
        "First-time Guest",
    )
    df["has_children"] = (df["children"] + df["babies"] > 0).astype(int)
    df["has_children_label"] = np.where(
        df["has_children"].eq(1),
        "With Children",
        "Without Children",
    )
    df["is_family"] = np.where(df["has_children"].eq(1), "Family", "Non-family")
    df["room_type_changed"] = (
        df["reserved_room_type"] != df["assigned_room_type"]
    ).astype(int)
    df["room_type_changed_label"] = np.where(
        df["room_type_changed"].eq(1),
        "Changed",
        "Same",
    )
    df["booking_season"] = df["arrival_date_month"].map(SEASON_MAP)
    df["arrival_quarter"] = "Q" + df["arrival_date"].dt.quarter.astype(str)
    df["arrival_year_month"] = df["arrival_date"].dt.to_period("M").astype(str)
    df["weekend_share"] = df["stays_in_weekend_nights"] / df["total_nights"]
    df["estimated_revenue"] = df["adr"] * df["total_nights"]
    df["realized_revenue"] = np.where(
        df["is_canceled"].eq(0),
        df["estimated_revenue"],
        0.0,
    )
    df["special_request_group"] = np.where(
        df["total_of_special_requests"].gt(0),
        "At least one request",
        "No request",
    )

    df["length_of_stay_category"] = pd.cut(
        df["total_nights"],
        bins=[0, 2, 4, 7, 14, np.inf],
        labels=["1-2 nights", "3-4 nights", "5-7 nights", "8-14 nights", "15+ nights"],
        right=True,
    )
    df["lead_time_segment"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, np.inf],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"],
        right=True,
    )
    df["adr_segment"] = pd.cut(
        df["adr"],
        bins=[-0.01, 50, 100, 150, 250, np.inf],
        labels=["0-50", "51-100", "101-150", "151-250", "250+"],
        right=True,
    )

    outlier_columns = [
        "lead_time",
        "adr",
        "total_nights",
        "total_guests",
        "days_in_waiting_list",
        "previous_cancellations",
    ]
    outlier_records: list[dict[str, Any]] = []
    for column in outlier_columns:
        bounds = calculate_iqr_bounds(df[column])
        outlier_mask = df[column].lt(bounds["lower"]) | df[column].gt(bounds["upper"])
        df[f"{column}_iqr_outlier"] = outlier_mask.astype(int)
        outlier_records.append(
            {
                "column": column,
                "q1": bounds["q1"],
                "q3": bounds["q3"],
                "iqr": bounds["iqr"],
                "lower_bound": bounds["lower"],
                "upper_bound": bounds["upper"],
                "outlier_count": int(outlier_mask.sum()),
                "outlier_rate": float(outlier_mask.mean()),
            }
        )

    outlier_summary = pd.DataFrame(outlier_records)
    adr_bounds = outlier_summary.loc[outlier_summary["column"].eq("adr")].iloc[0]
    lead_time_bounds = outlier_summary.loc[
        outlier_summary["column"].eq("lead_time")
    ].iloc[0]
    df["adr_capped"] = df["adr"].clip(
        lower=max(0, float(adr_bounds["lower_bound"])),
        upper=float(adr_bounds["upper_bound"]),
    )
    df["lead_time_capped"] = df["lead_time"].clip(
        lower=max(0, float(lead_time_bounds["lower_bound"])),
        upper=float(lead_time_bounds["upper_bound"]),
    )
    df["estimated_revenue_capped"] = df["adr_capped"] * df["total_nights"]
    df["realized_revenue_capped"] = np.where(
        df["is_canceled"].eq(0),
        df["estimated_revenue_capped"],
        0.0,
    )
    outlier_flag_columns = [f"{column}_iqr_outlier" for column in outlier_columns]
    df["any_iqr_outlier"] = df[outlier_flag_columns].max(axis=1)

    category_columns = [
        "hotel",
        "arrival_date_month",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "deposit_type",
        "agent",
        "company",
        "customer_type",
        "reservation_status",
        "is_canceled_label",
        "is_repeated_guest_label",
        "has_children_label",
        "is_family",
        "room_type_changed_label",
        "booking_season",
        "arrival_quarter",
        "length_of_stay_category",
        "lead_time_segment",
        "adr_segment",
        "special_request_group",
    ]
    for column in category_columns:
        df[column] = df[column].astype("category")

    cleaning_summary = {
        "source_url": DATA_URL,
        "raw_rows": int(len(raw_df)),
        "raw_columns": int(raw_df.shape[1]),
        "duplicates_removed": duplicates_before,
        "missing_cells_before": int(raw_df.isna().sum().sum()),
        "missing_cells_after": int(df.isna().sum().sum()),
        "invalid_guest_rows_removed": invalid_guest_rows,
        "invalid_night_rows_removed": invalid_night_rows,
        "negative_adr_rows_removed": negative_adr_rows,
        "invalid_date_rows_removed": invalid_date_rows,
        "clean_rows": int(len(df)),
        "clean_columns": int(df.shape[1]),
        "rows_removed_total": int(len(raw_df) - len(df)),
    }

    return df, cleaning_summary, outlier_summary, missing_before


def save_plot(plot_object: Any, filename: str) -> Path:
    """Save a Matplotlib figure or Seaborn grid to the Images directory."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / filename
    plot_object.savefig(path, bbox_inches="tight")
    plt.close("all")
    return path


def add_bar_labels(ax: plt.Axes, fmt: str = "{:,.0f}") -> None:
    """Add compact labels on a vertical bar chart."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=8)


def outlier_summary_to_markdown(outlier_summary: pd.DataFrame) -> str:
    """Convert the outlier summary to a markdown table without optional deps."""
    rows = "\n".join(
        "| {column} | {q1:.2f} | {q3:.2f} | {lower_bound:.2f} | "
        "{upper_bound:.2f} | {outlier_count:,} | {outlier_rate:.1%} |".format(
            **row
        )
        for row in outlier_summary.to_dict("records")
    )
    return (
        "| Column | Q1 | Q3 | Lower Bound | Upper Bound | Outlier Count | "
        "Outlier Rate |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{rows}"
    )


def plot_missing_values_bar(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    missing = raw_df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, palette="crest", hue=missing.index, legend=False)
    ax.set_title("Missing Values Before Cleaning")
    ax.set_xlabel("Missing Rows")
    ax.set_ylabel("Column")
    for index, value in enumerate(missing.values):
        ax.text(value, index, f" {value:,.0f}", va="center", fontsize=9)
    return fig


def plot_duplicate_records_bar(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    summary = pd.DataFrame(
        {
            "Stage": ["Raw rows", "Exact duplicates", "Cleaned rows"],
            "Rows": [
                cleaning_summary["raw_rows"],
                cleaning_summary["duplicates_removed"],
                cleaning_summary["clean_rows"],
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="Stage", y="Rows", ax=ax, palette="mako", hue="Stage", legend=False)
    ax.set_title("Dataset Size Before and After Cleaning")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    add_bar_labels(ax)
    return fig


def plot_cancellation_distribution(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.countplot(data=df, x="is_canceled_label", ax=ax, palette="Set2", hue="is_canceled_label", legend=False)
    ax.set_title("Booking Cancellation Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Bookings")
    add_bar_labels(ax)
    return fig


def plot_hotel_type_distribution(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.countplot(data=df, x="hotel", ax=ax, palette="Set1", hue="hotel", legend=False)
    ax.set_title("Booking Volume by Hotel Type")
    ax.set_xlabel("")
    ax.set_ylabel("Bookings")
    add_bar_labels(ax)
    return fig


def plot_cancellation_rate_by_hotel(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    rate = (
        df.groupby("hotel", observed=True)["is_canceled"]
        .mean()
        .mul(100)
        .reset_index(name="Cancellation Rate")
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=rate, x="hotel", y="Cancellation Rate", ax=ax, palette="rocket", hue="hotel", legend=False)
    ax.set_title("Cancellation Rate by Hotel Type")
    ax.set_xlabel("")
    ax.set_ylabel("Cancellation Rate (%)")
    add_bar_labels(ax, fmt="{:.1f}")
    return fig


def plot_monthly_bookings_trend(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    monthly = (
        df.groupby("arrival_year_month", observed=True)
        .size()
        .reset_index(name="Bookings")
        .sort_values("arrival_year_month")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="arrival_year_month", y="Bookings", marker="o", ax=ax)
    ax.set_title("Monthly Booking Trend")
    ax.set_xlabel("Arrival Month")
    ax.set_ylabel("Bookings")
    ax.tick_params(axis="x", rotation=60)
    return fig


def plot_monthly_cancellation_rate(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    monthly = (
        df.groupby(["arrival_month_number", "arrival_date_month"], observed=True)["is_canceled"]
        .mean()
        .mul(100)
        .reset_index(name="Cancellation Rate")
        .sort_values("arrival_month_number")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=monthly,
        x="arrival_date_month",
        y="Cancellation Rate",
        marker="o",
        ax=ax,
        color="#c44e52",
    )
    ax.set_title("Cancellation Rate by Arrival Month")
    ax.set_xlabel("Arrival Month")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_booking_volume_by_month(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.countplot(
        data=df,
        x="arrival_date_month",
        order=MONTH_ORDER,
        ax=ax,
        palette="viridis",
        hue="arrival_date_month",
        legend=False,
    )
    ax.set_title("Booking Volume by Calendar Month")
    ax.set_xlabel("Arrival Month")
    ax.set_ylabel("Bookings")
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_bookings_by_year(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    yearly = (
        df.groupby("arrival_date_year", observed=True)
        .size()
        .reset_index(name="Bookings")
        .sort_values("arrival_date_year")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=yearly, x="arrival_date_year", y="Bookings", ax=ax, palette="deep", hue="arrival_date_year", legend=False)
    ax.set_title("Bookings by Arrival Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Bookings")
    add_bar_labels(ax)
    return fig


def plot_market_segment_bookings(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    segment = df["market_segment"].value_counts().reset_index()
    segment.columns = ["Market Segment", "Bookings"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=segment, y="Market Segment", x="Bookings", ax=ax, palette="flare", hue="Market Segment", legend=False)
    ax.set_title("Bookings by Market Segment")
    ax.set_xlabel("Bookings")
    ax.set_ylabel("")
    return fig


def plot_distribution_channel_bookings(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    channel = df["distribution_channel"].value_counts().reset_index()
    channel.columns = ["Distribution Channel", "Bookings"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=channel, x="Bookings", y="Distribution Channel", ax=ax, palette="crest", hue="Distribution Channel", legend=False)
    ax.set_title("Bookings by Distribution Channel")
    ax.set_xlabel("Bookings")
    ax.set_ylabel("")
    return fig


def plot_customer_type_pie(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    counts = df["customer_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set3", n_colors=len(counts))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Customer Type Share")
    return fig


def plot_top_countries(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    countries = df["country"].value_counts().head(10).reset_index()
    countries.columns = ["Country", "Bookings"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=countries, x="Bookings", y="Country", ax=ax, palette="mako", hue="Country", legend=False)
    ax.set_title("Top 10 Source Countries")
    ax.set_xlabel("Bookings")
    ax.set_ylabel("")
    return fig


def plot_adr_histogram(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="adr_capped", bins=45, kde=True, ax=ax, color="#4c72b0")
    ax.set_title("ADR Distribution After IQR Capping")
    ax.set_xlabel("Average Daily Rate (ADR, capped for visualization)")
    ax.set_ylabel("Bookings")
    return fig


def plot_lead_time_distribution(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="lead_time_capped", bins=45, kde=True, ax=ax, color="#55a868")
    ax.set_title("Lead Time Distribution After IQR Capping")
    ax.set_xlabel("Lead Time in Days (capped for visualization)")
    ax.set_ylabel("Bookings")
    return fig


def plot_adr_boxplot_by_hotel(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="hotel", y="adr_capped", ax=ax, palette="pastel", hue="hotel", legend=False)
    ax.set_title("ADR Box Plot by Hotel Type")
    ax.set_xlabel("")
    ax.set_ylabel("ADR (capped for visualization)")
    return fig


def plot_lead_time_boxplot_by_status(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="is_canceled_label",
        y="lead_time_capped",
        ax=ax,
        palette="Set2",
        hue="is_canceled_label",
        legend=False,
    )
    ax.set_title("Lead Time Box Plot by Cancellation Status")
    ax.set_xlabel("")
    ax.set_ylabel("Lead Time in Days (capped)")
    return fig


def plot_adr_violin_by_customer_type(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df,
        x="customer_type",
        y="adr_capped",
        ax=ax,
        palette="Set3",
        hue="customer_type",
        legend=False,
        cut=0,
    )
    ax.set_title("ADR Violin Plot by Customer Type")
    ax.set_xlabel("Customer Type")
    ax.set_ylabel("ADR (capped)")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_total_nights_distribution(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    subset = df.loc[df["total_nights"].le(20)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=subset, x="total_nights", discrete=True, stat="count", ax=ax, color="#8172b2")
    ax.set_title("Distribution Plot of Length of Stay")
    ax.set_xlabel("Total Nights")
    ax.set_ylabel("Bookings")
    return fig


def plot_correlation_heatmap(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    numeric_columns = [
        "is_canceled",
        "lead_time",
        "adr_capped",
        "total_nights",
        "total_guests",
        "previous_cancellations",
        "booking_changes",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "room_type_changed",
        "realized_revenue_capped",
    ]
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    return fig


def plot_pair_plot_numeric_features(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> sns.axisgrid.PairGrid:
    columns = [
        "lead_time_capped",
        "adr_capped",
        "total_nights",
        "total_guests",
        "is_canceled_label",
    ]
    sample = df[columns].sample(min(1800, len(df)), random_state=42)
    grid = sns.pairplot(
        sample,
        hue="is_canceled_label",
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.35, "s": 18, "edgecolor": "none"},
        height=2.4,
    )
    grid.fig.suptitle("Pair Plot of Key Numeric Features", y=1.03, fontweight="bold")
    return grid


def plot_lead_time_vs_adr_scatter(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    sample = df.sample(min(6000, len(df)), random_state=7)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sample,
        x="lead_time_capped",
        y="adr_capped",
        hue="is_canceled_label",
        alpha=0.35,
        s=24,
        ax=ax,
    )
    ax.set_title("Lead Time vs ADR by Cancellation Status")
    ax.set_xlabel("Lead Time in Days (capped)")
    ax.set_ylabel("ADR (capped)")
    ax.legend(title="Status")
    return fig


def plot_realized_revenue_by_hotel(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    hotel_revenue = (
        df.groupby("hotel", observed=True)["realized_revenue_capped"]
        .sum()
        .reset_index(name="Realized Revenue")
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=hotel_revenue, x="hotel", y="Realized Revenue", ax=ax, palette="Set1", hue="hotel", legend=False)
    ax.set_title("Estimated Realized Room Revenue by Hotel Type")
    ax.set_xlabel("")
    ax.set_ylabel("ADR-units")
    add_bar_labels(ax)
    return fig


def plot_room_type_demand(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    room_types = df["reserved_room_type"].value_counts().head(10).reset_index()
    room_types.columns = ["Reserved Room Type", "Bookings"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=room_types,
        x="Bookings",
        y="Reserved Room Type",
        ax=ax,
        palette="cubehelix",
        hue="Reserved Room Type",
        legend=False,
    )
    ax.set_title("Top Reserved Room Types")
    ax.set_xlabel("Bookings")
    ax.set_ylabel("Room Type")
    return fig


def plot_special_requests_vs_cancellation(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    requests = (
        df.groupby("total_of_special_requests", observed=True)["is_canceled"]
        .mean()
        .mul(100)
        .reset_index(name="Cancellation Rate")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=requests,
        x="total_of_special_requests",
        y="Cancellation Rate",
        ax=ax,
        palette="Spectral",
        hue="total_of_special_requests",
        legend=False,
    )
    ax.set_title("Cancellation Rate by Number of Special Requests")
    ax.set_xlabel("Special Requests")
    ax.set_ylabel("Cancellation Rate (%)")
    add_bar_labels(ax, fmt="{:.1f}")
    return fig


def plot_seasonal_booking_patterns(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> plt.Figure:
    seasonal = (
        df.groupby(["booking_season", "hotel"], observed=True)
        .size()
        .reset_index(name="Bookings")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=seasonal,
        x="booking_season",
        y="Bookings",
        hue="hotel",
        order=SEASON_ORDER,
        ax=ax,
        palette="Set2",
    )
    ax.set_title("Seasonal Booking Patterns by Hotel Type")
    ax.set_xlabel("Season")
    ax.set_ylabel("Bookings")
    ax.legend(title="Hotel")
    return fig


CHART_REGISTRY: list[dict[str, Any]] = [
    {
        "filename": "01_missing_values_bar.png",
        "title": "Missing Values Before Cleaning",
        "type": "Bar Chart",
        "function": plot_missing_values_bar,
    },
    {
        "filename": "02_duplicate_records_bar.png",
        "title": "Dataset Size Before and After Cleaning",
        "type": "Bar Chart",
        "function": plot_duplicate_records_bar,
    },
    {
        "filename": "03_cancellation_distribution.png",
        "title": "Booking Cancellation Distribution",
        "type": "Count Plot",
        "function": plot_cancellation_distribution,
    },
    {
        "filename": "04_hotel_type_distribution.png",
        "title": "Booking Volume by Hotel Type",
        "type": "Count Plot",
        "function": plot_hotel_type_distribution,
    },
    {
        "filename": "05_cancellation_rate_by_hotel.png",
        "title": "Cancellation Rate by Hotel Type",
        "type": "Bar Chart",
        "function": plot_cancellation_rate_by_hotel,
    },
    {
        "filename": "06_monthly_bookings_trend.png",
        "title": "Monthly Booking Trend",
        "type": "Line Chart",
        "function": plot_monthly_bookings_trend,
    },
    {
        "filename": "07_monthly_cancellation_rate.png",
        "title": "Cancellation Rate by Arrival Month",
        "type": "Line Chart",
        "function": plot_monthly_cancellation_rate,
    },
    {
        "filename": "08_booking_volume_by_month.png",
        "title": "Booking Volume by Calendar Month",
        "type": "Count Plot",
        "function": plot_booking_volume_by_month,
    },
    {
        "filename": "09_bookings_by_year.png",
        "title": "Bookings by Arrival Year",
        "type": "Bar Chart",
        "function": plot_bookings_by_year,
    },
    {
        "filename": "10_market_segment_bookings.png",
        "title": "Bookings by Market Segment",
        "type": "Bar Chart",
        "function": plot_market_segment_bookings,
    },
    {
        "filename": "11_distribution_channel_bookings.png",
        "title": "Bookings by Distribution Channel",
        "type": "Bar Chart",
        "function": plot_distribution_channel_bookings,
    },
    {
        "filename": "12_customer_type_pie.png",
        "title": "Customer Type Share",
        "type": "Pie Chart",
        "function": plot_customer_type_pie,
    },
    {
        "filename": "13_top_countries.png",
        "title": "Top 10 Source Countries",
        "type": "Bar Chart",
        "function": plot_top_countries,
    },
    {
        "filename": "14_adr_histogram.png",
        "title": "ADR Distribution After IQR Capping",
        "type": "Histogram and Distribution Plot",
        "function": plot_adr_histogram,
    },
    {
        "filename": "15_lead_time_distribution.png",
        "title": "Lead Time Distribution After IQR Capping",
        "type": "Histogram and Distribution Plot",
        "function": plot_lead_time_distribution,
    },
    {
        "filename": "16_adr_boxplot_by_hotel.png",
        "title": "ADR Box Plot by Hotel Type",
        "type": "Box Plot",
        "function": plot_adr_boxplot_by_hotel,
    },
    {
        "filename": "17_lead_time_boxplot_by_status.png",
        "title": "Lead Time Box Plot by Cancellation Status",
        "type": "Box Plot",
        "function": plot_lead_time_boxplot_by_status,
    },
    {
        "filename": "18_adr_violin_by_customer_type.png",
        "title": "ADR Violin Plot by Customer Type",
        "type": "Violin Plot",
        "function": plot_adr_violin_by_customer_type,
    },
    {
        "filename": "19_total_nights_distribution.png",
        "title": "Distribution Plot of Length of Stay",
        "type": "Distribution Plot",
        "function": plot_total_nights_distribution,
    },
    {
        "filename": "20_correlation_heatmap.png",
        "title": "Correlation Matrix Heatmap",
        "type": "Heatmap and Correlation Matrix",
        "function": plot_correlation_heatmap,
    },
    {
        "filename": "21_pair_plot_numeric_features.png",
        "title": "Pair Plot of Key Numeric Features",
        "type": "Pair Plot",
        "function": plot_pair_plot_numeric_features,
    },
    {
        "filename": "22_lead_time_vs_adr_scatter.png",
        "title": "Lead Time vs ADR by Cancellation Status",
        "type": "Scatter Plot",
        "function": plot_lead_time_vs_adr_scatter,
    },
    {
        "filename": "23_realized_revenue_by_hotel.png",
        "title": "Estimated Realized Room Revenue by Hotel Type",
        "type": "Bar Chart",
        "function": plot_realized_revenue_by_hotel,
    },
    {
        "filename": "24_room_type_demand.png",
        "title": "Top Reserved Room Types",
        "type": "Bar Chart",
        "function": plot_room_type_demand,
    },
    {
        "filename": "25_special_requests_vs_cancellation.png",
        "title": "Cancellation Rate by Number of Special Requests",
        "type": "Bar Chart",
        "function": plot_special_requests_vs_cancellation,
    },
    {
        "filename": "26_seasonal_booking_patterns.png",
        "title": "Seasonal Booking Patterns by Hotel Type",
        "type": "Grouped Bar Chart",
        "function": plot_seasonal_booking_patterns,
    },
]


def generate_visualizations(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    outlier_summary: pd.DataFrame,
) -> list[Path]:
    """Generate and save all project visualizations."""
    saved_paths = []
    for chart in CHART_REGISTRY:
        plot_object = chart["function"](raw_df, df, cleaning_summary, outlier_summary)
        saved_paths.append(save_plot(plot_object, chart["filename"]))
    return saved_paths


def compute_business_metrics(df: pd.DataFrame, cleaning_summary: dict[str, Any]) -> dict[str, Any]:
    """Compute reusable metrics for README, notebook, and reports."""
    hotel_counts = df["hotel"].value_counts()
    cancellation_rate = float(df["is_canceled"].mean())
    hotel_cancel_rate = df.groupby("hotel", observed=True)["is_canceled"].mean()
    hotel_revenue = df.groupby("hotel", observed=True)["realized_revenue_capped"].sum()
    month_counts = df["arrival_date_month"].value_counts()
    monthly_cancel = df.groupby("arrival_date_month", observed=True)["is_canceled"].mean()
    yearly_counts = df.groupby("arrival_date_year", observed=True).size()
    market_counts = df["market_segment"].value_counts()
    channel_counts = df["distribution_channel"].value_counts()
    customer_counts = df["customer_type"].value_counts()
    customer_revenue = df.groupby("customer_type", observed=True)["realized_revenue_capped"].sum()
    country_counts = df["country"].value_counts()
    top_countries = country_counts.head(10).index
    top_country_cancel = (
        df.loc[df["country"].isin(top_countries)]
        .groupby("country", observed=True)["is_canceled"]
        .mean()
    )
    repeated_cancel = df.groupby("is_repeated_guest_label", observed=True)["is_canceled"].mean()
    request_cancel = df.groupby("special_request_group", observed=True)["is_canceled"].mean()
    lead_cancel = df.groupby("is_canceled_label", observed=True)["lead_time"].median()
    room_counts = df["reserved_room_type"].value_counts()
    room_change_rate = float(df["room_type_changed"].mean())
    room_change_cancel = df.groupby("room_type_changed_label", observed=True)["is_canceled"].mean()
    season_counts = df["booking_season"].value_counts()
    season_revenue = df.groupby("booking_season", observed=True)["realized_revenue_capped"].sum()
    length_counts = df["length_of_stay_category"].value_counts()
    special_request_counts = df["total_of_special_requests"].value_counts().sort_index()
    family_cancel = df.groupby("is_family", observed=True)["is_canceled"].mean()
    deposit_cancel = df.groupby("deposit_type", observed=True)["is_canceled"].mean()
    adr_by_hotel = df.groupby("hotel", observed=True)["adr_capped"].median()
    total_revenue = float(df["realized_revenue_capped"].sum())

    top_hotel = str(hotel_counts.idxmax())
    highest_cancel_hotel = str(hotel_cancel_rate.idxmax())
    top_month = str(month_counts.idxmax())
    highest_cancel_month = str(monthly_cancel.idxmax())
    top_year = int(yearly_counts.idxmax())
    top_market_segment = str(market_counts.idxmax())
    top_channel = str(channel_counts.idxmax())
    top_customer_type = str(customer_counts.idxmax())
    top_revenue_customer_type = str(customer_revenue.idxmax())
    top_country = str(country_counts.idxmax())
    highest_cancel_top_country = str(top_country_cancel.idxmax())
    top_room_type = str(room_counts.idxmax())
    top_season = str(season_counts.idxmax())
    top_revenue_season = str(season_revenue.idxmax())
    top_length_category = str(length_counts.idxmax())
    highest_deposit_cancel = str(deposit_cancel.idxmax())
    top_revenue_hotel = str(hotel_revenue.idxmax())
    highest_adr_hotel = str(adr_by_hotel.idxmax())

    return {
        "cleaning_summary": cleaning_summary,
        "overall_cancellation_rate": cancellation_rate,
        "hotel_counts": hotel_counts.astype(int).to_dict(),
        "hotel_cancel_rate": hotel_cancel_rate.astype(float).to_dict(),
        "hotel_revenue": hotel_revenue.astype(float).to_dict(),
        "monthly_cancel_rate": monthly_cancel.astype(float).to_dict(),
        "yearly_counts": yearly_counts.astype(int).to_dict(),
        "market_counts": market_counts.astype(int).to_dict(),
        "channel_counts": channel_counts.astype(int).to_dict(),
        "customer_counts": customer_counts.astype(int).to_dict(),
        "country_counts_top10": country_counts.head(10).astype(int).to_dict(),
        "top_country_cancel_rate": top_country_cancel.astype(float).to_dict(),
        "repeated_cancel_rate": repeated_cancel.astype(float).to_dict(),
        "request_cancel_rate": request_cancel.astype(float).to_dict(),
        "median_lead_time_by_status": lead_cancel.astype(float).to_dict(),
        "room_change_rate": room_change_rate,
        "room_change_cancel_rate": room_change_cancel.astype(float).to_dict(),
        "season_counts": season_counts.astype(int).to_dict(),
        "season_revenue": season_revenue.astype(float).to_dict(),
        "length_counts": length_counts.astype(int).to_dict(),
        "special_request_counts": special_request_counts.astype(int).to_dict(),
        "family_cancel_rate": family_cancel.astype(float).to_dict(),
        "deposit_cancel_rate": deposit_cancel.astype(float).to_dict(),
        "adr_by_hotel_median": adr_by_hotel.astype(float).to_dict(),
        "total_realized_revenue": total_revenue,
        "top_hotel": top_hotel,
        "top_hotel_count": int(hotel_counts.loc[top_hotel]),
        "highest_cancel_hotel": highest_cancel_hotel,
        "highest_cancel_hotel_rate": float(hotel_cancel_rate.loc[highest_cancel_hotel]),
        "top_revenue_hotel": top_revenue_hotel,
        "top_revenue_hotel_value": float(hotel_revenue.loc[top_revenue_hotel]),
        "highest_adr_hotel": highest_adr_hotel,
        "highest_adr_hotel_median": float(adr_by_hotel.loc[highest_adr_hotel]),
        "top_month": top_month,
        "top_month_count": int(month_counts.loc[top_month]),
        "highest_cancel_month": highest_cancel_month,
        "highest_cancel_month_rate": float(monthly_cancel.loc[highest_cancel_month]),
        "top_year": top_year,
        "top_year_count": int(yearly_counts.loc[top_year]),
        "top_market_segment": top_market_segment,
        "top_market_segment_count": int(market_counts.loc[top_market_segment]),
        "top_channel": top_channel,
        "top_channel_count": int(channel_counts.loc[top_channel]),
        "top_customer_type": top_customer_type,
        "top_customer_type_count": int(customer_counts.loc[top_customer_type]),
        "top_revenue_customer_type": top_revenue_customer_type,
        "top_revenue_customer_type_value": float(customer_revenue.loc[top_revenue_customer_type]),
        "top_country": top_country,
        "top_country_count": int(country_counts.loc[top_country]),
        "highest_cancel_top_country": highest_cancel_top_country,
        "highest_cancel_top_country_rate": float(
            top_country_cancel.loc[highest_cancel_top_country]
        ),
        "top_room_type": top_room_type,
        "top_room_type_count": int(room_counts.loc[top_room_type]),
        "top_season": top_season,
        "top_season_count": int(season_counts.loc[top_season]),
        "top_revenue_season": top_revenue_season,
        "top_revenue_season_value": float(season_revenue.loc[top_revenue_season]),
        "top_length_category": top_length_category,
        "top_length_category_count": int(length_counts.loc[top_length_category]),
        "highest_deposit_cancel": highest_deposit_cancel,
        "highest_deposit_cancel_rate": float(deposit_cancel.loc[highest_deposit_cancel]),
    }


def build_business_questions(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Build the business-question section from computed metrics."""
    request_cancel = metrics["request_cancel_rate"]
    no_request_rate = request_cancel.get("No request", 0.0)
    with_request_rate = request_cancel.get("At least one request", 0.0)
    repeated_cancel = metrics["repeated_cancel_rate"]
    first_time_rate = repeated_cancel.get("First-time Guest", 0.0)
    repeated_rate = repeated_cancel.get("Repeated Guest", 0.0)
    family_cancel = metrics["family_cancel_rate"]
    family_rate = family_cancel.get("Family", 0.0)
    non_family_rate = family_cancel.get("Non-family", 0.0)
    lead_time = metrics["median_lead_time_by_status"]
    changed_cancel = metrics["room_change_cancel_rate"].get("Changed", 0.0)
    same_cancel = metrics["room_change_cancel_rate"].get("Same", 0.0)

    return [
        {
            "question": "Which hotel type receives the most bookings?",
            "answer": (
                f"{metrics['top_hotel']} leads with {number(metrics['top_hotel_count'])} "
                "clean bookings."
            ),
        },
        {
            "question": "What is the overall cancellation rate?",
            "answer": (
                f"The cleaned dataset has an overall cancellation rate of "
                f"{percent(metrics['overall_cancellation_rate'])}."
            ),
        },
        {
            "question": "Which hotel type has the highest cancellation risk?",
            "answer": (
                f"{metrics['highest_cancel_hotel']} has the higher cancellation rate at "
                f"{percent(metrics['highest_cancel_hotel_rate'])}."
            ),
        },
        {
            "question": "Which hotel type contributes the most realized revenue?",
            "answer": (
                f"{metrics['top_revenue_hotel']} contributes the most estimated realized "
                f"room revenue: {revenue(metrics['top_revenue_hotel_value'])}."
            ),
        },
        {
            "question": "Which month has the highest booking demand?",
            "answer": (
                f"{metrics['top_month']} has the highest booking volume with "
                f"{number(metrics['top_month_count'])} bookings."
            ),
        },
        {
            "question": "Which month has the highest cancellation rate?",
            "answer": (
                f"{metrics['highest_cancel_month']} has the highest cancellation rate at "
                f"{percent(metrics['highest_cancel_month_rate'])}."
            ),
        },
        {
            "question": "Which year has the highest booking volume?",
            "answer": (
                f"{metrics['top_year']} has the most records with "
                f"{number(metrics['top_year_count'])} bookings."
            ),
        },
        {
            "question": "Which market segment is the main acquisition source?",
            "answer": (
                f"{metrics['top_market_segment']} is the largest segment with "
                f"{number(metrics['top_market_segment_count'])} bookings."
            ),
        },
        {
            "question": "Which distribution channel brings the most bookings?",
            "answer": (
                f"{metrics['top_channel']} is the leading distribution channel with "
                f"{number(metrics['top_channel_count'])} bookings."
            ),
        },
        {
            "question": "Which customer type dominates demand?",
            "answer": (
                f"{metrics['top_customer_type']} customers dominate with "
                f"{number(metrics['top_customer_type_count'])} bookings."
            ),
        },
        {
            "question": "Which customer type contributes the most realized revenue?",
            "answer": (
                f"{metrics['top_revenue_customer_type']} contributes the highest estimated "
                f"realized revenue: {revenue(metrics['top_revenue_customer_type_value'])}."
            ),
        },
        {
            "question": "Which country is the strongest source market?",
            "answer": (
                f"{metrics['top_country']} is the top country with "
                f"{number(metrics['top_country_count'])} bookings."
            ),
        },
        {
            "question": "Among top source countries, where is cancellation risk highest?",
            "answer": (
                f"{metrics['highest_cancel_top_country']} has the highest cancellation rate "
                f"among the top source countries at "
                f"{percent(metrics['highest_cancel_top_country_rate'])}."
            ),
        },
        {
            "question": "Do longer lead times relate to cancellation behavior?",
            "answer": (
                "Canceled bookings have a median lead time of "
                f"{decimal(lead_time.get('Canceled', 0.0))} days versus "
                f"{decimal(lead_time.get('Not Canceled', 0.0))} days for non-canceled bookings."
            ),
        },
        {
            "question": "Are repeated guests less likely to cancel?",
            "answer": (
                f"Repeated guests cancel at {percent(repeated_rate)}, while first-time guests "
                f"cancel at {percent(first_time_rate)}."
            ),
        },
        {
            "question": "Do special requests indicate stronger booking intent?",
            "answer": (
                f"Bookings with at least one special request cancel at {percent(with_request_rate)} "
                f"compared with {percent(no_request_rate)} for bookings with no request."
            ),
        },
        {
            "question": "Which reserved room type is most common?",
            "answer": (
                f"Room type {metrics['top_room_type']} is most requested with "
                f"{number(metrics['top_room_type_count'])} bookings."
            ),
        },
        {
            "question": "How often do room assignments change, and does it matter?",
            "answer": (
                f"Room type changes occur in {percent(metrics['room_change_rate'])} of bookings. "
                f"Changed-room bookings cancel at {percent(changed_cancel)} versus "
                f"{percent(same_cancel)} when the room type stays the same."
            ),
        },
        {
            "question": "Which season is busiest?",
            "answer": (
                f"{metrics['top_season']} is the busiest season with "
                f"{number(metrics['top_season_count'])} bookings."
            ),
        },
        {
            "question": "Which season contributes the most estimated realized revenue?",
            "answer": (
                f"{metrics['top_revenue_season']} contributes the highest estimated realized "
                f"revenue: {revenue(metrics['top_revenue_season_value'])}."
            ),
        },
        {
            "question": "What length of stay is most common?",
            "answer": (
                f"{metrics['top_length_category']} is the most common stay length category "
                f"with {number(metrics['top_length_category_count'])} bookings."
            ),
        },
        {
            "question": "Do families cancel differently from non-family bookings?",
            "answer": (
                f"Family bookings cancel at {percent(family_rate)}, while non-family bookings "
                f"cancel at {percent(non_family_rate)}."
            ),
        },
        {
            "question": "Which deposit type has the highest cancellation rate?",
            "answer": (
                f"{metrics['highest_deposit_cancel']} bookings show the highest cancellation "
                f"rate at {percent(metrics['highest_deposit_cancel_rate'])}."
            ),
        },
    ]


def build_chart_explanations(metrics: dict[str, Any]) -> dict[str, str]:
    """Create concise interpretation text for each visualization."""
    return {
        "01_missing_values_bar.png": (
            "Company and agent identifiers account for most missing values, so they are "
            "treated as explicit 'No Company' and 'No Agent' categories instead of being "
            "dropped. Country and children have small missing counts and are imputed safely."
        ),
        "02_duplicate_records_bar.png": (
            f"The project starts with {number(metrics['cleaning_summary']['raw_rows'])} rows "
            f"and removes {number(metrics['cleaning_summary']['duplicates_removed'])} exact "
            f"duplicates, leaving {number(metrics['cleaning_summary']['clean_rows'])} clean rows."
        ),
        "03_cancellation_distribution.png": (
            f"The overall cancellation rate is {percent(metrics['overall_cancellation_rate'])}, "
            "so cancellation is a major operational and revenue-risk theme in the dataset."
        ),
        "04_hotel_type_distribution.png": (
            f"{metrics['top_hotel']} has the larger booking base with "
            f"{number(metrics['top_hotel_count'])} records, making it the primary volume driver."
        ),
        "05_cancellation_rate_by_hotel.png": (
            f"{metrics['highest_cancel_hotel']} has the higher cancellation rate at "
            f"{percent(metrics['highest_cancel_hotel_rate'])}, which suggests different risk "
            "management strategies by hotel type."
        ),
        "06_monthly_bookings_trend.png": (
            "The monthly trend shows demand moving unevenly through time, with visible peaks "
            "that can support staffing, pricing, and campaign timing decisions."
        ),
        "07_monthly_cancellation_rate.png": (
            f"{metrics['highest_cancel_month']} has the highest cancellation rate at "
            f"{percent(metrics['highest_cancel_month_rate'])}, making it a priority month for "
            "deposit, reminder, or overbooking policies."
        ),
        "08_booking_volume_by_month.png": (
            f"{metrics['top_month']} is the strongest demand month with "
            f"{number(metrics['top_month_count'])} bookings."
        ),
        "09_bookings_by_year.png": (
            f"{metrics['top_year']} has the highest observed booking count. Because the dataset "
            "does not cover every month equally across all years, year comparisons should be "
            "interpreted as dataset-period trends."
        ),
        "10_market_segment_bookings.png": (
            f"{metrics['top_market_segment']} is the leading acquisition segment with "
            f"{number(metrics['top_market_segment_count'])} bookings."
        ),
        "11_distribution_channel_bookings.png": (
            f"{metrics['top_channel']} is the main distribution channel with "
            f"{number(metrics['top_channel_count'])} bookings."
        ),
        "12_customer_type_pie.png": (
            f"{metrics['top_customer_type']} customers represent the biggest share of demand, "
            "so retention and cancellation controls should be tailored to this group."
        ),
        "13_top_countries.png": (
            f"{metrics['top_country']} is the strongest source market with "
            f"{number(metrics['top_country_count'])} bookings."
        ),
        "14_adr_histogram.png": (
            "ADR is right-skewed, so the project keeps the original values but uses IQR-capped "
            "ADR for readable visuals and robust revenue comparisons."
        ),
        "15_lead_time_distribution.png": (
            "Lead time is also right-skewed, showing that most bookings are made within a "
            "moderate window while a smaller group books far in advance."
        ),
        "16_adr_boxplot_by_hotel.png": (
            f"{metrics['highest_adr_hotel']} has the higher median capped ADR at "
            f"{decimal(metrics['highest_adr_hotel_median'])}, indicating stronger pricing power."
        ),
        "17_lead_time_boxplot_by_status.png": (
            "Canceled bookings tend to have longer lead times, which makes lead time a useful "
            "early warning feature for cancellation-risk monitoring."
        ),
        "18_adr_violin_by_customer_type.png": (
            "The violin plot shows how ADR distributions differ by customer type, helping "
            "separate high-volume customer groups from high-rate customer groups."
        ),
        "19_total_nights_distribution.png": (
            f"{metrics['top_length_category']} is the most common stay-length category, which "
            "is useful for room inventory planning and housekeeping schedules."
        ),
        "20_correlation_heatmap.png": (
            "The correlation matrix highlights directional relationships: special requests "
            "move opposite to cancellation, while lead time is positively associated with risk."
        ),
        "21_pair_plot_numeric_features.png": (
            "The pair plot shows feature interactions and confirms that canceled and non-canceled "
            "bookings overlap heavily, so business rules should combine several indicators."
        ),
        "22_lead_time_vs_adr_scatter.png": (
            "The scatter plot shows cancellation status across pricing and lead-time space, "
            "surfacing long-lead bookings as a visible risk cluster."
        ),
        "23_realized_revenue_by_hotel.png": (
            f"{metrics['top_revenue_hotel']} produces the largest estimated realized room "
            f"revenue at {revenue(metrics['top_revenue_hotel_value'])}."
        ),
        "24_room_type_demand.png": (
            f"Reserved room type {metrics['top_room_type']} is the dominant room product, "
            "so availability and pricing decisions around this type carry high impact."
        ),
        "25_special_requests_vs_cancellation.png": (
            "Cancellation rate declines as special requests increase, suggesting requests are "
            "a practical signal of guest commitment."
        ),
        "26_seasonal_booking_patterns.png": (
            f"{metrics['top_season']} is the busiest season, while "
            f"{metrics['top_revenue_season']} contributes the most estimated realized revenue."
        ),
    }


def write_data_dictionary(df: pd.DataFrame, outlier_summary: pd.DataFrame) -> None:
    """Write a professional data dictionary markdown report."""
    original_columns = [
        ("hotel", "Hotel type: City Hotel or Resort Hotel."),
        ("is_canceled", "Cancellation flag. 1 means canceled, 0 means not canceled."),
        ("lead_time", "Days between booking date and arrival date."),
        ("arrival_date_year", "Arrival year."),
        ("arrival_date_month", "Arrival month name."),
        ("arrival_date_week_number", "Arrival ISO week number."),
        ("arrival_date_day_of_month", "Arrival day of month."),
        ("stays_in_weekend_nights", "Number of weekend nights booked."),
        ("stays_in_week_nights", "Number of week nights booked."),
        ("adults", "Number of adults."),
        ("children", "Number of children."),
        ("babies", "Number of babies."),
        ("meal", "Meal package type."),
        ("country", "Guest country of origin."),
        ("market_segment", "Market segment designation."),
        ("distribution_channel", "Booking distribution channel."),
        ("is_repeated_guest", "Repeated guest flag."),
        ("previous_cancellations", "Previous canceled bookings by the guest."),
        (
            "previous_bookings_not_canceled",
            "Previous non-canceled bookings by the guest.",
        ),
        ("reserved_room_type", "Reserved room category code."),
        ("assigned_room_type", "Assigned room category code."),
        ("booking_changes", "Number of changes made to the booking."),
        ("deposit_type", "Deposit policy type."),
        ("agent", "Travel agency identifier."),
        ("company", "Company/entity identifier."),
        ("days_in_waiting_list", "Days the booking stayed on the waiting list."),
        ("customer_type", "Customer contract/category type."),
        ("adr", "Average Daily Rate, calculated per occupied room night."),
        (
            "required_car_parking_spaces",
            "Number of required parking spaces.",
        ),
        (
            "total_of_special_requests",
            "Number of special requests made by the guest.",
        ),
        ("reservation_status", "Final reservation status."),
        ("reservation_status_date", "Date of final reservation status."),
    ]
    engineered_columns = [
        ("arrival_date", "Parsed arrival date."),
        ("arrival_month_number", "Numeric arrival month."),
        ("total_nights", "Weekend nights plus week nights."),
        ("total_guests", "Adults plus children plus babies."),
        ("is_canceled_label", "Human-readable cancellation status."),
        ("is_repeated_guest_label", "Human-readable repeated guest status."),
        ("has_children", "Binary flag for bookings with children or babies."),
        ("has_children_label", "Human-readable child presence label."),
        ("is_family", "Family/non-family booking flag."),
        ("room_type_changed", "Flag for reserved room type differing from assigned type."),
        ("room_type_changed_label", "Human-readable room assignment change label."),
        ("booking_season", "Season inferred from arrival month."),
        ("arrival_quarter", "Quarter inferred from arrival date."),
        ("arrival_year_month", "Year-month period used for trend charts."),
        ("weekend_share", "Share of stay nights that fall on weekends."),
        ("estimated_revenue", "ADR multiplied by total nights."),
        (
            "realized_revenue",
            "Estimated revenue retained only for non-canceled bookings.",
        ),
        ("length_of_stay_category", "Binned length-of-stay segment."),
        ("lead_time_segment", "Binned lead-time segment."),
        ("adr_segment", "Binned ADR segment."),
        ("adr_capped", "ADR capped at IQR bounds for robust visualization."),
        (
            "lead_time_capped",
            "Lead time capped at IQR bounds for robust visualization.",
        ),
        (
            "estimated_revenue_capped",
            "IQR-capped ADR multiplied by total nights.",
        ),
        (
            "realized_revenue_capped",
            "Capped revenue proxy retained only for non-canceled bookings.",
        ),
        ("any_iqr_outlier", "Flag for any selected numeric IQR outlier."),
    ]
    outlier_rows = "\n".join(
        "| {column} | {q1:.2f} | {q3:.2f} | {lower_bound:.2f} | "
        "{upper_bound:.2f} | {outlier_count:,} | {outlier_rate:.1%} |".format(
            **row
        )
        for row in outlier_summary.to_dict("records")
    )
    original_rows = "\n".join(
        f"| `{column}` | {description} | `{df[column].dtype if column in df else 'raw only'}` |"
        for column, description in original_columns
    )
    engineered_rows = "\n".join(
        f"| `{column}` | {description} | `{df[column].dtype}` |"
        for column, description in engineered_columns
        if column in df.columns
    )
    content = f"""# Data Dictionary

Dataset: Hotel Booking Demand

Source CSV: {DATA_URL}

## Original Columns

| Column | Description | Cleaned dtype |
|---|---|---|
{original_rows}

## Engineered Columns

| Column | Description | Cleaned dtype |
|---|---|---|
{engineered_rows}

## IQR Outlier Summary

Outliers are flagged using the standard IQR rule:

`lower_bound = Q1 - 1.5 * IQR`

`upper_bound = Q3 + 1.5 * IQR`

| Column | Q1 | Q3 | Lower Bound | Upper Bound | Outlier Count | Outlier Rate |
|---|---:|---:|---:|---:|---:|---:|
{outlier_rows}
"""
    (REPORT_DIR / "Data_Dictionary.md").write_text(content, encoding="utf-8")


def write_business_report(
    metrics: dict[str, Any],
    business_questions: list[dict[str, str]],
    chart_explanations: dict[str, str],
    outlier_summary: pd.DataFrame,
) -> None:
    """Write the main business insights report."""
    questions_md = "\n".join(
        f"{index}. **{item['question']}**\n\n   {item['answer']}"
        for index, item in enumerate(business_questions, start=1)
    )
    chart_md = "\n".join(
        f"### {index}. {chart['title']}\n\n"
        f"- Visualization type: {chart['type']}\n"
        f"- Screenshot: `Images/{chart['filename']}`\n"
        f"- Insight: {chart_explanations[chart['filename']]}\n"
        for index, chart in enumerate(CHART_REGISTRY, start=1)
    )
    outlier_md = outlier_summary_to_markdown(outlier_summary)
    content = f"""# Business Insights Report

## Executive Summary

This EDA project analyzes {number(metrics['cleaning_summary']['clean_rows'])} cleaned hotel
booking records from a real public dataset. The business focus is demand patterns,
cancellation risk, source-market behavior, customer segmentation, room demand, and an
estimated room-revenue proxy calculated as `ADR x total_nights` for non-canceled bookings.

Key headline findings:

- Overall cancellation rate: **{percent(metrics['overall_cancellation_rate'])}**
- Highest-volume hotel type: **{metrics['top_hotel']}**
- Highest-risk hotel type: **{metrics['highest_cancel_hotel']}**
- Strongest demand month: **{metrics['top_month']}**
- Largest market segment: **{metrics['top_market_segment']}**
- Strongest source country: **{metrics['top_country']}**
- Most common room type: **{metrics['top_room_type']}**

## Business Questions Answered

{questions_md}

## Visualization-by-Visualization Insights

{chart_md}

## Outlier Detection Summary

The project uses IQR outlier flags for numeric fields that are important for hotel
operations and revenue analysis. Outliers are not blindly removed because extreme lead
times, long stays, and high ADR bookings can be real business events. For readable
visuals, ADR and lead time are capped only in visualization-specific fields.

{outlier_md}

## Recommended Actions

1. Use lead time, deposit type, special requests, and customer type as early cancellation
   risk signals.
2. Build segment-specific retention playbooks for the dominant market segment and
   customer type.
3. Protect high-demand months with tighter inventory controls and cancellation policies.
4. Monitor source countries with high cancellation risk among top-volume markets.
5. Use room type demand and room assignment changes to improve inventory allocation.
"""
    (REPORT_DIR / "Business_Insights_Report.md").write_text(
        content,
        encoding="utf-8",
    )


def write_final_conclusion(metrics: dict[str, Any]) -> None:
    """Write the final conclusion report."""
    content = f"""# Final Conclusion

The Hotel Booking Demand EDA shows that cancellation management is the central business
problem in this dataset. After cleaning, the project retains
{number(metrics['cleaning_summary']['clean_rows'])} valid booking records, comfortably
above the 20,000-record requirement.

The analysis finds that **{metrics['top_hotel']}** drives the most booking volume, while
**{metrics['highest_cancel_hotel']}** carries the higher cancellation rate. Demand is
seasonal, with **{metrics['top_month']}** standing out as the strongest booking month and
**{metrics['top_season']}** as the busiest season. The dominant acquisition engine is
**{metrics['top_market_segment']}**, and **{metrics['top_country']}** is the largest source
country.

From a revenue perspective, **{metrics['top_revenue_hotel']}** contributes the most
estimated realized room revenue. Operationally, lead time, deposit type, repeated guest
status, special requests, room type changes, and customer type all provide useful signals
for cancellation-risk monitoring.

This project is suitable for a Data Analyst internship portfolio because it demonstrates
real-data cleaning, missing-value treatment, duplicate handling, type correction, feature
engineering, outlier detection with IQR, business-question framing, and professional
visual storytelling with Matplotlib and Seaborn.
"""
    (REPORT_DIR / "Final_Conclusion.md").write_text(content, encoding="utf-8")


def write_requirements() -> None:
    """Write reproducible Python requirements."""
    content = """pandas==3.0.3
numpy==2.5.0
matplotlib==3.11.0
seaborn==0.13.2
scipy==1.18.0
nbformat==5.10.4
nbconvert==7.17.1
jupyter==1.1.1
ipykernel==7.3.0
"""
    (PROJECT_ROOT / "requirements.txt").write_text(content, encoding="utf-8")


def write_gitignore() -> None:
    """Write Git ignore rules for a clean portfolio repository."""
    content = """.DS_Store
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
venv/
env/
.env
*.log
.matplotlib_cache/
.ipython/
.jupyter/
"""
    (PROJECT_ROOT / ".gitignore").write_text(content, encoding="utf-8")


def write_license() -> None:
    """Write an MIT license for the project code."""
    content = """MIT License

Copyright (c) 2026 Sujal Giri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Dataset rights belong to the original dataset authors and source publishers.
"""
    (PROJECT_ROOT / "LICENSE").write_text(content, encoding="utf-8")


def write_readme(
    metrics: dict[str, Any],
    business_questions: list[dict[str, str]],
    chart_explanations: dict[str, str],
) -> None:
    """Write a GitHub-ready README with all chart screenshots."""
    question_rows = "\n".join(
        f"| {index} | {item['question']} | {item['answer']} |"
        for index, item in enumerate(business_questions, start=1)
    )
    chart_gallery = "\n\n".join(
        f"### {index}. {chart['title']}\n\n"
        f"**Type:** {chart['type']}\n\n"
        f"![{chart['title']}](Images/{chart['filename']})\n\n"
        f"**Insight:** {chart_explanations[chart['filename']]}"
        for index, chart in enumerate(CHART_REGISTRY, start=1)
    )
    tree = """EDA_Project/
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
|-- .gitignore"""
    content = f"""# Hotel Booking Demand - End-to-End Exploratory Data Analysis

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

- Dataset mirror used in this project: {DATA_URL}
- TidyTuesday source page: https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11
- Original Kaggle dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Original paper DOI: https://doi.org/10.1016/j.dib.2018.11.126

## Dataset Size

| Metric | Value |
|---|---:|
| Raw rows | {number(metrics['cleaning_summary']['raw_rows'])} |
| Raw columns | {metrics['cleaning_summary']['raw_columns']} |
| Exact duplicates removed | {number(metrics['cleaning_summary']['duplicates_removed'])} |
| Clean rows | {number(metrics['cleaning_summary']['clean_rows'])} |
| Clean columns | {metrics['cleaning_summary']['clean_columns']} |
| Minimum record requirement | 20,000+ |
| Requirement met | Yes |

## Project Structure

```text
{tree}
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

- Overall cancellation rate: **{percent(metrics['overall_cancellation_rate'])}**
- Highest-volume hotel type: **{metrics['top_hotel']}**
- Highest cancellation-risk hotel type: **{metrics['highest_cancel_hotel']}**
- Strongest booking month: **{metrics['top_month']}**
- Leading market segment: **{metrics['top_market_segment']}**
- Leading distribution channel: **{metrics['top_channel']}**
- Strongest source country: **{metrics['top_country']}**
- Most requested room type: **{metrics['top_room_type']}**
- Highest realized revenue hotel type: **{metrics['top_revenue_hotel']}**

## Business Questions Answered

| # | Business Question | Answer |
|---:|---|---|
{question_rows}

## Visualization Gallery and Insights

{chart_gallery}

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

## Author

**Sujal Giri**  
B.Tech CSE, 3rd Semester  
BS in Computer Science and Data Analytics, IIT Patna
"""
    (PROJECT_ROOT / "README.md").write_text(content, encoding="utf-8")


def create_notebook(
    metrics: dict[str, Any],
    business_questions: list[dict[str, str]],
    chart_explanations: dict[str, str],
) -> None:
    """Create a clean, executable Jupyter notebook."""
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    cells: list[Any] = []
    cells.append(
        nbf.v4.new_markdown_cell(
            """# Hotel Booking Demand - End-to-End Exploratory Data Analysis

This notebook is a professional portfolio EDA project using a real public dataset with
more than 20,000 records. It covers data loading, data cleaning, missing values,
duplicates, type correction, outlier detection, feature engineering, visualization,
business questions, and final conclusions."""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Dataset Source

- Source CSV: {DATA_URL}
- Original Kaggle dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Research paper DOI: https://doi.org/10.1016/j.dib.2018.11.126"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "Dataset").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.append(str(PROJECT_ROOT / "scripts"))
import hotel_booking_eda as hbeda

sns.set_theme(style="whitegrid", palette="deep")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 140)

print(f"Project root: {PROJECT_ROOT}")"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 1 - Load the Real Public Dataset

The CSV is stored locally in the `Dataset` folder so the notebook can run without
Kaggle authentication or live downloads."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """raw_df = hbeda.load_raw_data(PROJECT_ROOT / "Dataset" / "hotel_bookings.csv")
print(f"Raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")
raw_df.head()"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 2 - Initial Data Audit

Before cleaning, inspect missing values, duplicate rows, data types, and core numeric
summary statistics."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """audit_summary = pd.DataFrame({
    "dtype": raw_df.dtypes.astype(str),
    "missing_values": raw_df.isna().sum(),
    "missing_rate": raw_df.isna().mean().round(4),
    "unique_values": raw_df.nunique()
}).sort_values("missing_values", ascending=False)

print(f"Exact duplicate rows: {raw_df.duplicated().sum():,}")
audit_summary.head(12)"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """raw_df.describe(include="all").transpose().head(20)"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 3 - Data Cleaning and Feature Engineering

The reusable project function performs the full cleaning pipeline:

- Fills missing identifier/category values.
- Removes exact duplicates.
- Converts dates and corrects data types.
- Removes invalid zero-guest, zero-night, negative-ADR, and invalid-date rows.
- Engineers business features for stay length, guests, seasonality, revenue proxy,
  room changes, family bookings, and segmentation."""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """clean_df, cleaning_summary, outlier_summary, missing_before = hbeda.clean_data(raw_df)
clean_df.to_csv(PROJECT_ROOT / "Dataset" / "hotel_bookings_cleaned.csv", index=False)

pd.DataFrame([cleaning_summary]).transpose().rename(columns={0: "value"})"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 4 - IQR Outlier Detection

Outliers are detected using the standard IQR rule. They are flagged rather than blindly
deleted because extreme lead times, high ADR values, and long stays can represent real
hotel business events."""
        )
    )
    cells.append(nbf.v4.new_code_cell("outlier_summary"))
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 5 - Business Metrics"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            """metrics = hbeda.compute_business_metrics(clean_df, cleaning_summary)
business_questions = hbeda.build_business_questions(metrics)
pd.DataFrame(business_questions)"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 6 - Professional Visualizations

Each chart below is generated with Matplotlib and/or Seaborn, saved into the `Images`
folder, and interpreted from a business perspective."""
        )
    )

    for index, chart in enumerate(CHART_REGISTRY, start=1):
        cells.append(
            nbf.v4.new_markdown_cell(
                f"""### {index}. {chart['title']}

**Visualization type:** {chart['type']}

**Business insight:** {chart_explanations[chart['filename']]}"""
            )
        )
        function_name = chart["function"].__name__
        filename = chart["filename"]
        cells.append(
            nbf.v4.new_code_cell(
                f"""plot_object = hbeda.{function_name}(raw_df, clean_df, cleaning_summary, outlier_summary)
image_path = PROJECT_ROOT / "Images" / "{filename}"
plot_object.savefig(image_path, bbox_inches="tight")
plt.close("all")
display(Image(filename=str(image_path)))"""
            )
        )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Step 7 - Final Business Questions"""
        )
    )
    cells.append(nbf.v4.new_code_cell("pd.DataFrame(business_questions)"))
    conclusion_points = "\n".join(
        f"- {item['answer']}" for item in business_questions[:10]
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Final Conclusion

The analysis shows that hotel cancellation risk is strongly connected to booking behavior,
channel mix, lead time, special requests, deposit policy, and customer type. The project
retains {number(metrics['cleaning_summary']['clean_rows'])} clean records, uses robust
outlier handling, and translates exploratory findings into business actions.

Selected conclusions:

{conclusion_points}
"""
        )
    )

    nb["cells"] = cells
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")


def write_summary_json(
    metrics: dict[str, Any],
    business_questions: list[dict[str, str]],
    outlier_summary: pd.DataFrame,
) -> None:
    """Write machine-readable analysis summary."""
    payload = {
        "metrics": metrics,
        "business_questions": business_questions,
        "outlier_summary": outlier_summary.to_dict("records"),
        "chart_count": len(CHART_REGISTRY),
        "charts": [
            {
                "filename": chart["filename"],
                "title": chart["title"],
                "type": chart["type"],
            }
            for chart in CHART_REGISTRY
        ],
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_project_build() -> None:
    """Run the complete portfolio project build."""
    ensure_directories()
    raw_df = load_raw_data()
    clean_df, cleaning_summary, outlier_summary, _ = clean_data(raw_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    metrics = compute_business_metrics(clean_df, cleaning_summary)
    business_questions = build_business_questions(metrics)
    chart_explanations = build_chart_explanations(metrics)

    generate_visualizations(raw_df, clean_df, cleaning_summary, outlier_summary)
    write_data_dictionary(clean_df, outlier_summary)
    write_business_report(metrics, business_questions, chart_explanations, outlier_summary)
    write_final_conclusion(metrics)
    write_requirements()
    write_gitignore()
    write_license()
    write_readme(metrics, business_questions, chart_explanations)
    create_notebook(metrics, business_questions, chart_explanations)
    write_summary_json(metrics, business_questions, outlier_summary)

    print("EDA project build completed.")
    print(f"Cleaned dataset rows: {cleaning_summary['clean_rows']:,}")
    print(f"Charts generated: {len(CHART_REGISTRY)}")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"README: {PROJECT_ROOT / 'README.md'}")


if __name__ == "__main__":
    run_project_build()
