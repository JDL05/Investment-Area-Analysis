import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---- PAGE CONFIG: FULL WIDTH ----
st.set_page_config(layout="wide", page_title="TVF Territory Dashboard")

# Use Seaborn with a consistent "Blues" theme
sns.set_theme(style="whitegrid")

#####################################################################
#                    TERRITORY DATA MERGING                         #
#####################################################################

territory_files = [
    "Territory_2020_2025.csv",
    "Territory_2018_2019.csv",
    "Territory_2017.csv",
    "Territory_2016.csv"
]

@st.cache_data
def merge_territory_files():
    """Merge multiple Territory CSV files into one."""
    missing_files = [file for file in territory_files if not os.path.exists(file)]
    if missing_files:
        st.error(f"The following files are missing: {missing_files}")
        st.stop()

    dataframes = [pd.read_csv(file) for file in territory_files]
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    combined_df = combined_df.drop_duplicates()

    output_file = "Territory_2016_2025.csv"
    combined_df.to_csv(output_file, index=False)
    return output_file

@st.cache_data
def load_territory_data(path: str) -> pd.DataFrame:
    """Load and process the merged Territory dataset."""
    df = pd.read_csv(path)

    # Convert 'Founded Date' to datetime
    df["Founded Date"] = pd.to_datetime(df["Founded Date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Founded Date"])

    # Extract year
    df["Year Founded"] = df["Founded Date"].dt.year

    # Clean up 'Total Funding Amount' to numeric (EUR)
    if "Total Funding Amount" in df.columns:
        df["Total Funding Amount"] = (
            df["Total Funding Amount"]
            .replace(r"[^0-9.]", "", regex=True)
            .replace("", float("nan"))
            .astype(float, errors="ignore")
        )
    return df

#####################################################################
#                       FUNDING ROUNDS DATA                         #
#####################################################################

@st.cache_data
def load_funding_data() -> pd.DataFrame:
    """
    Load the 'FundingRounds_2016_2025.csv' file, unify city names (Köln/Cologne -> 'Cologne'),
    and keep only the city portion before the first comma.
    """
    funding_file = "FundingRounds_2016_2025.csv"
    if not os.path.exists(funding_file):
        st.warning(f"{funding_file} not found in directory.")
        return pd.DataFrame()

    df = pd.read_csv(funding_file)

    # Convert 'Announced Date' to datetime
    if "Announced Date" in df.columns:
        df["Announced Date"] = pd.to_datetime(df["Announced Date"], errors="coerce")
        df = df.dropna(subset=["Announced Date"])
        df["Year Announced"] = df["Announced Date"].dt.year

    # "Money Raised" -> float
    if "Money Raised" in df.columns:
        df["Money Raised"] = (
            df["Money Raised"]
            .replace(r"[^0-9.]", "", regex=True)
            .replace("", float("nan"))
            .astype(float, errors="ignore")
        )
    else:
        df["Money Raised"] = float("nan")

    if "Funding Type" not in df.columns:
        df["Funding Type"] = "Unknown"

    # Normalize city names in "Organization Location"
    if "Organization Location" in df.columns:
        def unify_city(location_str):
            if pd.isna(location_str):
                return location_str
            parts = location_str.split(",")
            if not parts:
                return location_str
            city = parts[0].strip()
            if city.lower() in ["köln", "cologne"]:
                city = "Cologne"
            return city

        df["Organization Location"] = df["Organization Location"].fillna("").apply(unify_city)

    return df

#####################################################################
#                           MAIN LOGIC                              #
#####################################################################

# Merge and load territory data
merged_territory_file = merge_territory_files()
territory_df = load_territory_data(merged_territory_file)
funding_df = load_funding_data()

# Ensure territory data exists
if territory_df.empty or territory_df["Year Founded"].isna().all():
    st.error("No valid 'Territory' data found. Please check your CSV files.")
    st.stop()

# Determine min/max years for a unified slider
all_territory_years = territory_df["Year Founded"].dropna().unique()
all_funding_years = funding_df["Year Announced"].dropna().unique() if not funding_df.empty else []
combined_years = list(set(all_territory_years).union(set(all_funding_years)))
if not combined_years:
    st.error("No valid years found in either dataset.")
    st.stop()

min_year = int(min(combined_years))
max_year = int(max(combined_years))

# Slider in the sidebar
st.sidebar.title("Select Year Range (Unified Filter)")
selected_year_range = st.sidebar.slider(
    "Choose a range of years",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter territory & funding
filtered_territory = territory_df[
    territory_df["Year Founded"].between(selected_year_range[0], selected_year_range[1])
]
filtered_funding = (
    funding_df[
        funding_df["Year Announced"].between(selected_year_range[0], selected_year_range[1])
    ]
    if not funding_df.empty
    else pd.DataFrame()
)

# Title
st.title("TVF Territory")

#####################################################################
#                    TERRITORY DASHBOARD SECTIONS                   #
#####################################################################

st.subheader("Territory Data")

# --- METRIC TILES (COLORFUL) ---
total_companies = filtered_territory.shape[0]
if "Headquarters Location" in filtered_territory.columns:
    is_aachen = filtered_territory["Headquarters Location"].str.contains(
        "Aachen, Nordrhein-Westfalen, Germany",
        case=False,
        na=False
    )
    aachen_count = is_aachen.sum()
else:
    aachen_count = 0

# Most common Industry Group
if "Industry Groups" in filtered_territory.columns and not filtered_territory.empty:
    industry_series = filtered_territory["Industry Groups"].dropna().astype(str)
    all_groups = []
    for line in industry_series:
        all_groups.extend([grp.strip() for grp in line.split(",")])
    if all_groups:
        most_common_group, _ = Counter(all_groups).most_common(1)[0]
    else:
        most_common_group = "N/A"
else:
    most_common_group = "N/A"

# Average Funding (in EUR)
if "Total Funding Amount" in filtered_territory.columns and not filtered_territory.empty:
    valid_funding = filtered_territory["Total Funding Amount"].dropna()
    if not valid_funding.empty:
        avg_funding_val = valid_funding.mean()
        avg_funding_str = f"{avg_funding_val:,.0f} EUR"
    else:
        avg_funding_str = "0 EUR"
else:
    avg_funding_str = "0 EUR"

# Define some tile colors
tile_colors = ["#822723", "#822723", "#822723", "#822723"]  # green, blue, orange, purple

# Define a helper function to display smaller font tiles
def colored_tile(column, label, value, bg_color):
    """
    Renders a colorful 'tile' in the given column using custom HTML,
    but with smaller font sizes.
    """
    column.markdown(f"""
    <div style="background-color:{bg_color}; padding:10px; border-radius:10px; text-align:center;">
        <h4 style="color:white; margin:0; font-size:14px;">{label}</h4>
        <p style="color:white; font-size:18px; font-weight:bold; margin:0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
colored_tile(col1, "Total Companies", f"{total_companies}", tile_colors[0])
colored_tile(col2, "Companies in Aachen", f"{aachen_count}", tile_colors[1])
colored_tile(col3, "Most Common Industry Group", f"{most_common_group}", tile_colors[2])
colored_tile(col4, "Average Funding", avg_funding_str, tile_colors[3])

st.write("---")

# Territory Charts
col_left, col_right = st.columns(2)

col_left.subheader("Number of Startups Founded (Per Year)")
if filtered_territory.empty:
    col_left.warning("No territory data in the selected year range.")
else:
    year_counts = filtered_territory["Year Founded"].value_counts().sort_index()
    if year_counts.empty:
        col_left.warning("No data found.")
    else:
        data_for_plot = pd.DataFrame({"Year": year_counts.index, "Count": year_counts.values})
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x="Year",
            y="Count",
            data=data_for_plot,
            palette="Blues",
            edgecolor="black",
            ax=ax
        )
        for i, row in data_for_plot.iterrows():
            ax.text(
                i,
                row["Count"] + 0.5,
                str(int(row["Count"])),
                ha="center",
                va="bottom",
                color="black",
                fontsize=9
            )
        ax.set_title(f"({selected_year_range[0]} - {selected_year_range[1]})", fontsize=12)
        ax.set_xlabel("Year Founded", fontsize=10)
        ax.set_ylabel("Number of Companies", fontsize=10)
        ax.set_ylim(0, data_for_plot["Count"].max() * 1.1)
        col_left.pyplot(fig)

col_right.subheader("Top 10 Industries")
if "Industry Groups" not in filtered_territory.columns or filtered_territory.empty:
    col_right.warning("No industry data in the selected range.")
else:
    industry_series = filtered_territory["Industry Groups"].dropna().astype(str)
    all_groups = []
    for line in industry_series:
        all_groups.extend([grp.strip() for grp in line.split(",")])
    if not all_groups:
        col_right.warning("No valid industry data.")
    else:
        top10 = Counter(all_groups).most_common(10)
        top10_df = pd.DataFrame(top10, columns=["Industry", "Count"]).sort_values("Count", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            y="Industry",
            x="Count",
            data=top10_df,
            palette="Blues",
            edgecolor="black",
            ax=ax2
        )
        for i, row in top10_df.iterrows():
            ax2.text(
                row["Count"] + 0.3,
                i,
                str(int(row["Count"])),
                va="center",
                color="black",
                fontsize=9
            )
        ax2.set_title("Most Frequent Industries", fontsize=12)
        ax2.set_xlabel("Count", fontsize=10)
        ax2.set_ylabel("")
        ax2.set_ylim(-0.5, len(top10_df) - 0.5)
        col_right.pyplot(fig2)

st.write("---")

#####################################################################
#                     FUNDING ROUNDS DASHBOARD                      #
#####################################################################

st.subheader("Funding Rounds Data")

if filtered_funding.empty:
    st.warning("No funding rounds data in the selected year range.")
else:
    total_rounds = filtered_funding.shape[0]
    if not filtered_funding["Money Raised"].dropna().empty:
        avg_money_val = filtered_funding["Money Raised"].dropna().mean()
        avg_money_str = f"{avg_money_val:,.0f} EUR"
    else:
        avg_money_str = "0 EUR"

    if "Funding Type" in filtered_funding.columns and not filtered_funding["Funding Type"].dropna().empty:
        top_funding_type = filtered_funding["Funding Type"].value_counts().index[0]
    else:
        top_funding_type = "N/A"

    # Another set of tile colors for funding data
    funding_tile_colors = ["#822723", "#822723", "#822723"]  # pink, teal, amber

    colA, colB, colC = st.columns(3)
    colored_tile(colA, "Total Funding Rounds", total_rounds, funding_tile_colors[0])
    colored_tile(colB, "Average Money Raised", avg_money_str, funding_tile_colors[1])
    colored_tile(colC, "Most Common Funding Type", top_funding_type, funding_tile_colors[2])

    st.write("---")

    left_fr, right_fr = st.columns(2)

    left_fr.write("#### Funding Rounds by Year")
    if "Year Announced" in filtered_funding.columns:
        year_counts_fr = filtered_funding["Year Announced"].value_counts().sort_index()
        if year_counts_fr.empty:
            left_fr.warning("No valid 'Year Announced' data in this range.")
        else:
            data_fr_plot = pd.DataFrame({"Year": year_counts_fr.index, "Rounds": year_counts_fr.values})
            fig_fr, ax_fr = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x="Year",
                y="Rounds",
                data=data_fr_plot,
                palette="Blues",
                edgecolor="black",
                ax=ax_fr
            )
            for i, row in data_fr_plot.iterrows():
                ax_fr.text(
                    i,
                    row["Rounds"] + 0.3,
                    str(int(row["Rounds"])),
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=9
                )
            ax_fr.set_title(f"Funding Rounds by Year ({selected_year_range[0]} - {selected_year_range[1]})", fontsize=11)
            ax_fr.set_xlabel("Year Announced", fontsize=9)
            ax_fr.set_ylabel("Number of Rounds", fontsize=9)
            ax_fr.set_ylim(0, data_fr_plot["Rounds"].max() * 1.1)

            left_fr.pyplot(fig_fr)
    else:
        left_fr.warning("No 'Year Announced' column found in the data.")

    right_fr.write("#### Funding Type Distribution")
    if "Funding Type" in filtered_funding.columns and not filtered_funding["Funding Type"].dropna().empty:
        ft_counts = filtered_funding["Funding Type"].value_counts()
        total_counts = ft_counts.sum()
        percent_series = ft_counts / total_counts
        main_slices = percent_series[percent_series >= 0.03]
        other_slices = percent_series[percent_series < 0.03]
        if other_slices.any():
            main_slices["Others"] = other_slices.sum()
        final_counts = (main_slices * total_counts).round().astype(int)
        colors_ft = sns.color_palette("Blues", n_colors=len(final_counts))

        fig_ft, ax_ft = plt.subplots(figsize=(4, 4))
        ax_ft.pie(
            final_counts.values,
            labels=final_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_ft
        )
        ax_ft.set_title("Funding Type", fontsize=11)
        ax_ft.axis("equal")
        right_fr.pyplot(fig_ft)
    else:
        right_fr.warning("No valid 'Funding Type' data in this range.")

    st.write("---")

    loc_col, table_col = st.columns(2)

    loc_col.subheader("Most Frequent Organization Location")
    if "Organization Location" not in filtered_funding.columns or filtered_funding["Organization Location"].dropna().empty:
        loc_col.warning("No 'Organization Location' data in this range.")
    else:
        loc_counts = filtered_funding["Organization Location"].value_counts().head(10)
        if loc_counts.empty:
            loc_col.warning("No valid Organization Location found.")
        else:
            loc_df = pd.DataFrame({"Location": loc_counts.index, "Count": loc_counts.values})

            fig_loc, ax_loc = plt.subplots(figsize=(6, 4))
            sns.barplot(
                y="Location",
                x="Count",
                data=loc_df,
                palette="Blues",
                edgecolor="black",
                ax=ax_loc
            )
            for i, row in loc_df.iterrows():
                ax_loc.text(
                    row["Count"] + 0.3,
                    i,
                    str(int(row["Count"])),
                    va="center",
                    color="black",
                    fontsize=9
                )
            ax_loc.set_title("Organization Location (Top 10)", fontsize=12)
            ax_loc.set_xlabel("Count", fontsize=10)
            ax_loc.set_ylabel("")
            loc_col.pyplot(fig_loc)

    table_col.subheader("10 Most Recent Funding Rounds")
    if "Announced Date" in filtered_funding.columns:
        recent_rounds = (
            filtered_funding.dropna(subset=["Announced Date"])
            .sort_values("Announced Date", ascending=False)
            .head(10)
        )
        columns_to_show = [
            "Organization Name",
            "Funding Type",
            "Organization Location",
            "Money Raised",
            "Announced Date"
        ]
        existing_cols = [c for c in columns_to_show if c in recent_rounds.columns]
        table_col.dataframe(recent_rounds[existing_cols])
    else:
        table_col.warning("No 'Announced Date' column found. Cannot show recent funding rounds.")

st.write("---")
st.sidebar.write("Use the year range slider to explore different time periods.")
