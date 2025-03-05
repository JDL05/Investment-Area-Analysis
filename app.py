import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import matplotlib.ticker as mticker  # For custom y-axis formatting in millions

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
    keep only the city portion before the first comma, etc.
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

    # Convert "Money Raised" to float
    if "Money Raised" in df.columns:
        df["Money Raised"] = (
            df["Money Raised"]
            .replace(r"[^0-9.]", "", regex=True)
            .replace("", float("nan"))
            .astype(float, errors="ignore")
        )
    else:
        df["Money Raised"] = float("nan")

    # Ensure "Funding Stage" column exists
    if "Funding Stage" not in df.columns:
        df["Funding Stage"] = "Unknown"

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

# ============ Font-size Controls ============
ANNOTATION_FONTSIZE = 7  # numeric labels inside bars
AXIS_LABEL_FONTSIZE = 6  # x/y axis label font size
TICK_LABEL_FONTSIZE = 6  # x/y axis tick label font size

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

# Sidebar slider
st.sidebar.title("Select Year Range")
selected_year_range = st.sidebar.slider(
    "Choose a range of years",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# 1) Filter territory data by year
filtered_territory = territory_df[
    territory_df["Year Founded"].between(selected_year_range[0], selected_year_range[1])
]

# 2) Filter funding data by year + stage
filtered_funding = funding_df[
    (funding_df["Year Announced"].between(selected_year_range[0], selected_year_range[1]))
    & (funding_df["Funding Stage"].isin(["Seed", "Early Stage Funding"]))
    ]

# Title
st.title("TVF Territory")

#####################################################################
#                    TERRITORY DASHBOARD SECTIONS                   #
#####################################################################

st.subheader("Territory Data")

# --- METRIC TILES ---
total_companies = filtered_territory.shape[0]

# Companies in Aachen
if "Headquarters Location" in filtered_territory.columns:
    is_aachen = filtered_territory["Headquarters Location"].str.contains(
        "Aachen, Nordrhein-Westfalen, Germany",
        case=False,
        na=False
    )
    aachen_count = is_aachen.sum()
else:
    aachen_count = 0

# Safely determine most common Industry Group
if "Industry Groups" in filtered_territory.columns and not filtered_territory.empty:
    industry_series = filtered_territory["Industry Groups"].dropna().astype(str)
    all_groups = []
    for line in industry_series:
        all_groups.extend([grp.strip() for grp in line.split(",")])

    if all_groups:
        top_item = Counter(all_groups).most_common(1)
        if top_item:
            most_common_group = top_item[0][0]
        else:
            most_common_group = "N/A"
    else:
        most_common_group = "N/A"
else:
    most_common_group = "N/A"

# Companies with Funding
if "Funding Status" in filtered_territory.columns and not filtered_territory.empty:
    has_funding_count = filtered_territory["Funding Status"].dropna().shape[0]
else:
    has_funding_count = 0

tile_colors = ["#822723", "#822723", "#822723", "#822723"]

def colored_tile(column, label, value, bg_color):
    """
    Renders a tile using HTML with smaller fonts.
    """
    column.markdown(f"""
    <div style="background-color:{bg_color}; padding:10px; border-radius:10px; text-align:center;">
        <h4 style="color:white; margin:0; font-size:14px;">{label}</h4>
        <p style="color:white; font-size:18px; font-weight:bold; margin:0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
colored_tile(col1, "Total Companies", f"{total_companies}", tile_colors[0])
colored_tile(col2, "Aachen Companies", f"{aachen_count}", tile_colors[1])
colored_tile(col3, "Top Industry", f"{most_common_group}", tile_colors[2])
colored_tile(col4, "Companies w/ Funding", f"{has_funding_count}", tile_colors[3])

st.write("---")

# Territory Charts
col_left, col_right = st.columns(2)

col_left.subheader("Startups by Year")
if filtered_territory.empty:
    col_left.warning("No territory data in range.")
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
        # Numeric labels
        for i, row in data_for_plot.iterrows():
            ax.text(
                i,
                row["Count"] + 0.5,
                str(int(row["Count"])),
                ha="center",
                va="bottom",
                color="black",
                fontsize=ANNOTATION_FONTSIZE
            )

        ax.set_xlabel("Year Founded", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax.set_ylim(0, data_for_plot["Count"].max() * 1.1)
        col_left.pyplot(fig)

col_right.subheader("Top Industries")
if "Industry Groups" not in filtered_territory.columns or filtered_territory.empty:
    col_right.warning("No industry data in range.")
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
        # Numeric labels
        for i, row in top10_df.iterrows():
            ax2.text(
                row["Count"] + 0.3,
                i,
                str(int(row["Count"])),
                va="center",
                color="black",
                fontsize=ANNOTATION_FONTSIZE
            )

        ax2.set_xlabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
        ax2.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)
        ax2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax2.set_ylim(-0.5, len(top10_df) - 0.5)
        col_right.pyplot(fig2)

st.write("---")

#####################################################################
#                     FUNDING ROUNDS DASHBOARD                      #
#####################################################################

st.subheader("Funding Rounds (Seed & Early)")

if filtered_funding.empty:
    st.warning("No seed/early rounds in range.")
else:
    # Basic tiles
    total_rounds = filtered_funding.shape[0]

    # Average Round Size
    if not filtered_funding["Money Raised"].dropna().empty:
        avg_money_val = filtered_funding["Money Raised"].dropna().mean()
        avg_money_str = f"{avg_money_val:,.0f} EUR"
    else:
        avg_money_str = "0 EUR"

    # Most Common Stage
    if "Funding Stage" in filtered_funding.columns and not filtered_funding["Funding Stage"].dropna().empty:
        top_stage = filtered_funding["Funding Stage"].value_counts().index[0]
    else:
        top_stage = "N/A"

    funding_tile_colors = ["#822723", "#822723", "#822723"]

    colA, colB, colC = st.columns(3)

    colored_tile(colA, "Total Rounds", total_rounds, funding_tile_colors[0])
    colored_tile(colB, "Avg Round Size", avg_money_str, funding_tile_colors[1])
    colored_tile(colC, "Top Stage", top_stage, funding_tile_colors[2])

    st.write("---")

    # Rounds by Year
    left_fr, right_fr = st.columns(2)
    left_fr.subheader("Rounds by Year")

    if "Year Announced" in filtered_funding.columns:
        year_counts_fr = filtered_funding["Year Announced"].value_counts().sort_index()
        if year_counts_fr.empty:
            left_fr.warning("No 'Year Announced' in range.")
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
                    fontsize=ANNOTATION_FONTSIZE
                )

            ax_fr.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
            ax_fr.set_ylabel("Rounds", fontsize=AXIS_LABEL_FONTSIZE)
            ax_fr.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
            ax_fr.set_ylim(0, data_fr_plot["Rounds"].max() * 1.1)
            left_fr.pyplot(fig_fr)
    else:
        left_fr.warning("No 'Year Announced' column found.")

    # Money by Year (DISPLAY IN MILLIONS)
    right_fr.subheader("Investments by Year")
    if "Year Announced" in filtered_funding.columns:
        sum_by_year = filtered_funding.groupby("Year Announced")["Money Raised"].sum().sort_index()
        if sum_by_year.empty:
            right_fr.warning("No 'Money Raised' in range.")
        else:
            fig_sum, ax_sum = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=sum_by_year.index.astype(int),
                y=sum_by_year.values,
                palette="Blues",
                edgecolor="black",
                ax=ax_sum
            )

            # Annotate each bar in millions
            for i, val in enumerate(sum_by_year.values):
                text_position = val + (val * 0.01 if val != 0 else 1)
                annotation_str = f"{val / 1_000_000:.1f}M"
                ax_sum.text(
                    i,
                    text_position,
                    annotation_str,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=ANNOTATION_FONTSIZE
                )

            # Label the axes
            ax_sum.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
            ax_sum.set_ylabel("EUR (millions)", fontsize=AXIS_LABEL_FONTSIZE)

            # Format the y-axis ticks in millions
            ax_sum.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: f"{x / 1_000_000:.0f}M")
            )

            # Adjust tick label font size
            ax_sum.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

            max_val = sum_by_year.max()
            ax_sum.set_ylim(0, max_val * 1.15 if max_val != 0 else 1)
            right_fr.pyplot(fig_sum)
    else:
        right_fr.warning("No 'Year Announced' or 'Money Raised' column.")

    st.write("---")

    # Location & Recent Rounds
    loc_col, table_col = st.columns(2)

    loc_col.subheader("Top Locations")
    if "Organization Location" not in filtered_funding.columns or filtered_funding["Organization Location"].dropna().empty:
        loc_col.warning("No location data in range.")
    else:
        loc_counts = filtered_funding["Organization Location"].value_counts().head(10)
        if loc_counts.empty:
            loc_col.warning("No locations found.")
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
                    fontsize=ANNOTATION_FONTSIZE
                )
            ax_loc.set_xlabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
            ax_loc.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)
            ax_loc.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
            loc_col.pyplot(fig_loc)

    table_col.subheader("Recent Rounds")
    if "Announced Date" in filtered_funding.columns:
        recent_rounds = (
            filtered_funding.dropna(subset=["Announced Date"])
            .sort_values("Announced Date", ascending=False)
            .head(10)
        )
        columns_to_show = [
            "Organization Name",
            "Funding Stage",
            "Organization Location",
            "Money Raised",
            "Announced Date",
            "Investor Names"
        ]
        existing_cols = [c for c in columns_to_show if c in recent_rounds.columns]
        table_col.dataframe(recent_rounds[existing_cols])
    else:
        table_col.warning("No 'Announced Date' to show recent rounds.")

    st.write("---")

    #####################################################################
    #               INVESTOR ANALYSIS (Seed/Early Stage)                #
    #####################################################################

    st.write("### Investor Analysis")

    if "Investor Names" not in filtered_funding.columns:
        st.warning("No 'Investor Names' column in dataset.")
    else:
        # Gather all investor names
        all_investors = []
        for val in filtered_funding["Investor Names"].dropna():
            investors_list = [inv.strip() for inv in val.split(",") if inv.strip()]
            all_investors.extend(investors_list)

        if not all_investors:
            st.warning("No valid investor names found.")
        else:
            colInv1, colInv2 = st.columns(2)

            # -------------------
            # Top Investors (Count)
            # -------------------
            with colInv1:
                st.subheader("Top Investors (Count)")
                investor_counts = Counter(all_investors)
                top10_investors = investor_counts.most_common(10)
                df_top10_count = pd.DataFrame(top10_investors, columns=["Investor", "Count"])

                fig_i1, ax_i1 = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=df_top10_count,
                    y="Investor",
                    x="Count",
                    palette="Blues",
                    edgecolor="black",
                    ax=ax_i1
                )
                for i, row in df_top10_count.iterrows():
                    ax_i1.text(
                        row["Count"] + 0.1,
                        i,
                        str(row["Count"]),
                        va="center",
                        color="black",
                        fontsize=ANNOTATION_FONTSIZE
                    )
                ax_i1.set_xlabel("Rounds", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i1.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i1.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
                colInv1.pyplot(fig_i1)

            # ------------------
            # Top Investors (EUR) in Millions
            # ------------------
            with colInv2:
                st.subheader("Top Investors (EUR)")
                investor_money = defaultdict(float)

                # Sum 'Money Raised' for each round among that row's investors
                for idx, row in filtered_funding.dropna(subset=["Investor Names"]).iterrows():
                    money_raised = row["Money Raised"] if not pd.isna(row["Money Raised"]) else 0.0
                    i_list = [inv.strip() for inv in row["Investor Names"].split(",") if inv.strip()]
                    for inv in i_list:
                        investor_money[inv] += money_raised

                # Sort and get top 10
                sorted_investors = sorted(investor_money.items(), key=lambda x: x[1], reverse=True)
                top10_by_money = sorted_investors[:10]
                df_top10_money = pd.DataFrame(top10_by_money, columns=["Investor", "TotalMoney"])

                fig_i2, ax_i2 = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=df_top10_money,
                    x="Investor",
                    y="TotalMoney",
                    palette="Blues",
                    edgecolor="black",
                    ax=ax_i2
                )

                # Annotate each bar in millions
                for i, row in df_top10_money.iterrows():
                    text_position = row["TotalMoney"] + (row["TotalMoney"] * 0.01 if row["TotalMoney"] != 0 else 1)
                    # Convert the numeric value to millions (e.g., "5.2M")
                    annotation_str = f"{row['TotalMoney'] / 1_000_000:.1f}M"
                    ax_i2.text(
                        i,
                        text_position,
                        annotation_str,
                        ha="center",
                        va="bottom",
                        color="black",
                        fontsize=ANNOTATION_FONTSIZE
                    )

                # Label the axes
                ax_i2.set_xlabel("Investor", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i2.set_ylabel("EUR (millions)", fontsize=AXIS_LABEL_FONTSIZE)

                # Format the y-axis ticks in millions
                ax_i2.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, pos: f"{x / 1_000_000:.0f}M")
                )

                # Rotate x-axis labels if needed
                plt.setp(ax_i2.get_xticklabels(), rotation=45, ha="right")

                # Adjust tick label font size
                ax_i2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

                colInv2.pyplot(fig_i2)

st.sidebar.write("Use the year range slider to explore different time periods.")
