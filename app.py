import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

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

    # Make sure there's a column called "Funding Stage" for your data
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

# A single variable to adjust the annotation font size everywhere:
ANNOTATION_FONTSIZE = 6  # <- Change this value to control the size of the numbers on all charts

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
st.sidebar.title("Select Year Range (Unified Filter)")
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

# 2) Filter funding data by year AND by Funding Stage in ["Seed", "Early Stage Funding"]
filtered_funding = funding_df[
    (funding_df["Year Announced"].between(selected_year_range[0], selected_year_range[1])) &
    (funding_df["Funding Stage"].isin(["Seed", "Early Stage Funding"]))
    ]

# Title
st.title("TVF Territory")

#####################################################################
#                    TERRITORY DASHBOARD SECTIONS                   #
#####################################################################

st.subheader("Territory Data")

# --- METRIC TILES (COLORFUL) ---
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

# Companies with Funding (simply checking if "Funding Status" has data)
if "Funding Status" in filtered_territory.columns and not filtered_territory.empty:
    has_funding_count = filtered_territory["Funding Status"].dropna().shape[0]
else:
    has_funding_count = 0

tile_colors = ["#822723", "#822723", "#822723", "#822723"]

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
colored_tile(col4, "Companies with Funding", f"{has_funding_count}", tile_colors[3])

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
                fontsize=ANNOTATION_FONTSIZE  # <--- Here is where we use the variable
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
                fontsize=ANNOTATION_FONTSIZE  # <--- Using the variable again
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

st.subheader("Funding Rounds Data (Only Seed & Early Stage Funding)")

# If filtered_funding is empty, it means no deals match year + Funding Stage filters
if filtered_funding.empty:
    st.warning("No Seed/Early Stage funding rounds in the selected year range.")
else:
    # 1) Basic Tiles
    total_rounds = filtered_funding.shape[0]

    # Average Round Size
    if not filtered_funding["Money Raised"].dropna().empty:
        avg_money_val = filtered_funding["Money Raised"].dropna().mean()
        avg_money_str = f"{avg_money_val:,.0f} EUR"
    else:
        avg_money_str = "0 EUR"

    # Because we only have "Seed" and "Early Stage Funding" in this subset,
    # the "Most Common Funding Stage" will just be one or the other, or might tie.
    if "Funding Stage" in filtered_funding.columns and not filtered_funding["Funding Stage"].dropna().empty:
        top_stage = filtered_funding["Funding Stage"].value_counts().index[0]
    else:
        top_stage = "N/A"

    funding_tile_colors = ["#822723", "#822723", "#822723"]

    colA, colB, colC = st.columns([1, 1, 1])

    def colored_tile(column, label, value, bg_color):
        column.markdown(f"""
        <div style="background-color:{bg_color}; padding:10px; border-radius:10px; text-align:center;">
            <h4 style="color:white; margin:0; font-size:14px;">{label}</h4>
            <p style="color:white; font-size:18px; font-weight:bold; margin:0;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

    colored_tile(colA, "Total Funding Rounds", total_rounds, funding_tile_colors[0])
    colored_tile(colB, "Average Round Size", avg_money_str, funding_tile_colors[1])
    colored_tile(colC, "Most Common Stage", top_stage, funding_tile_colors[2])

    st.write("---")

    # 2) Funding Rounds by Year
    left_fr, right_fr = st.columns(2)
    left_fr.write("#### Rounds by Year")

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
                    fontsize=ANNOTATION_FONTSIZE  # <--- Using the variable
                )
            ax_fr.set_title(f"Rounds by Year ({selected_year_range[0]} - {selected_year_range[1]})", fontsize=11)
            ax_fr.set_xlabel("Year Announced", fontsize=9)
            ax_fr.set_ylabel("Number of Rounds", fontsize=9)
            ax_fr.set_ylim(0, data_fr_plot["Rounds"].max() * 1.1)
            left_fr.pyplot(fig_fr)
    else:
        left_fr.warning("No 'Year Announced' column found in the data.")

    # 3) Sum of Money Raised by Year (still only Seed/Early Stage)
    right_fr.write("#### Total Money Raised by Year")
    if "Year Announced" in filtered_funding.columns:
        sum_by_year = filtered_funding.groupby("Year Announced")["Money Raised"].sum().sort_index()
        if sum_by_year.empty:
            right_fr.warning("No valid 'Money Raised' data in this range.")
        else:
            fig_sum, ax_sum = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x=sum_by_year.index.astype(int),
                y=sum_by_year.values,
                palette="Blues",
                edgecolor="black",
                ax=ax_sum
            )
            for i, val in enumerate(sum_by_year.values):
                ax_sum.text(
                    i,
                    val + (val * 0.01 if val != 0 else 1),
                    f"{val:,.0f}",
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=ANNOTATION_FONTSIZE  # <--- Using the variable
                )
            ax_sum.set_title("Total Money Raised by Year", fontsize=11)
            ax_sum.set_xlabel("Year", fontsize=9)
            ax_sum.set_ylabel("EUR (Sum of Money Raised)", fontsize=9)
            ax_sum.set_ylim(0, sum_by_year.max() * 1.15 if sum_by_year.max() != 0 else 1)
            right_fr.pyplot(fig_sum)
    else:
        right_fr.warning("No 'Year Announced' or 'Money Raised' columns found.")

    st.write("---")

    # 4) Organization Location & Recent Funding Rounds
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
                    fontsize=ANNOTATION_FONTSIZE  # <--- Using the variable
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
            "Funding Stage",
            "Organization Location",
            "Money Raised",
            "Announced Date",
            "Investor Names"
        ]
        existing_cols = [c for c in columns_to_show if c in recent_rounds.columns]
        table_col.dataframe(recent_rounds[existing_cols])
    else:
        table_col.warning("No 'Announced Date' column found. Cannot show recent funding rounds.")

    st.write("---")

    #####################################################################
    #               INVESTOR ANALYSIS (Seed/Early Stage)                #
    #####################################################################

    st.write("### Investor Analysis (Only Seed & Early Stage)")

    if "Investor Names" not in filtered_funding.columns:
        st.warning("No 'Investor Names' column found in the dataset.")
    else:
        # Gather all investor names across all rows, splitting by comma
        all_investors = []
        for val in filtered_funding["Investor Names"].dropna():
            investors_list = [inv.strip() for inv in val.split(",") if inv.strip()]
            all_investors.extend(investors_list)

        if not all_investors:
            st.warning("No valid investor names found in 'Investor Names' column.")
        else:
            # 1) TOP 10 MOST ACTIVE INVESTORS
            colInv1, colInv2 = st.columns(2)

            with colInv1:
                st.write("#### Top 10 Most Active Investors")
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
                        fontsize=ANNOTATION_FONTSIZE  # <--- Using the variable
                    )
                ax_i1.set_xlabel("Number of Rounds Participated")
                ax_i1.set_ylabel("")
                ax_i1.set_title("Most Active Investors (Seed/Early Stage)", fontsize=11)
                colInv1.pyplot(fig_i1)

            # 2) TOP 10 INVESTORS BY TOTAL MONEY RAISED
            with colInv2:
                st.write("#### Top 10 Investors by Total Money Raised")
                investor_money = defaultdict(float)

                # Sum 'Money Raised' for each round among the row's investors
                for idx, row in filtered_funding.dropna(subset=["Investor Names"]).iterrows():
                    money_raised = row["Money Raised"] if not pd.isna(row["Money Raised"]) else 0.0
                    i_list = [inv.strip() for inv in row["Investor Names"].split(",") if inv.strip()]
                    for inv in i_list:
                        investor_money[inv] += money_raised

                # Sort by total money descending, get top 10
                sorted_investors = sorted(
                    investor_money.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
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
                for i, row in df_top10_money.iterrows():
                    ax_i2.text(
                        i,
                        row["TotalMoney"] + (row["TotalMoney"] * 0.01 if row["TotalMoney"] != 0 else 1),
                        f"{row['TotalMoney']:,.0f}",
                        ha="center",
                        va="bottom",
                        color="black",
                        fontsize=ANNOTATION_FONTSIZE  # <--- Using the variable
                    )
                ax_i2.set_xlabel("Investor")
                ax_i2.set_ylabel("Sum of Money Raised")
                ax_i2.set_title("Top 10 Investors by Total Money (Seed/Early)", fontsize=11)
                plt.setp(ax_i2.get_xticklabels(), rotation=45, ha="right")
                colInv2.pyplot(fig_i2)

st.sidebar.write("Use the year range slider to explore different time periods.")
