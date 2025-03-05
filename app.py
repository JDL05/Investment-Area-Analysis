import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import matplotlib.ticker as mticker  # For custom y-axis formatting in millions
import numpy as np
import matplotlib.colors as mcolors

# ------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="TVF Territory Dashboard + Merged Funding")
# ------------------------------------------------------------------------------

################################################################################
#                               FILE LISTS                                     #
################################################################################

# TERRITORY (Companies) files to merge
territory_files = [
    "Territory_2020_2025.csv",
    "Territory_2018_2019.csv",
    "Territory_2017.csv",
    "Territory_2016.csv"
]

# FUNDING ROUNDS (Investments) files to merge
funding_rounds_files = [
    "FundingRounds_2016_2025.csv",
    "FundingRounds_2016_2025_BE.csv"
]

################################################################################
#                           MERGE FUNCTIONS                                    #
################################################################################

@st.cache_data
def merge_territory_files() -> str:
    """Merge multiple Territory CSV files into one, return the output CSV path."""
    missing_files = [file for file in territory_files if not os.path.exists(file)]
    if missing_files:
        st.error(f"The following Territory files are missing: {missing_files}")
        st.stop()

    dfs = [pd.read_csv(file) for file in territory_files]
    combined_df = pd.concat(dfs, ignore_index=True, sort=False).drop_duplicates()

    output_file = "Territory_2016_2025_merged.csv"
    combined_df.to_csv(output_file, index=False)
    return output_file


@st.cache_data
def merge_funding_rounds_files() -> str:
    missing_files = [file for file in funding_rounds_files if not os.path.exists(file)]
    if missing_files:
        st.error(f"The following Funding Rounds files are missing: {missing_files}")
        st.stop()

    dfs = [pd.read_csv(file) for file in funding_rounds_files]
    combined_df = pd.concat(dfs, ignore_index=True, sort=False).drop_duplicates()

    output_file = "FundingRounds_2016_2025_merged.csv"
    combined_df.to_csv(output_file, index=False)
    return output_file


################################################################################
#                           LOAD FUNCTIONS                                     #
################################################################################

@st.cache_data
def load_territory_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Territory file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df["Founded Date"] = pd.to_datetime(df["Founded Date"], format="%Y-%m-%d", errors="coerce")
    df.dropna(subset=["Founded Date"], inplace=True)

    df["Year Founded"] = df["Founded Date"].dt.year

    if "Total Funding Amount" in df.columns:
        df["Total Funding Amount"] = (
            df["Total Funding Amount"]
            .replace(r"[^0-9.]", "", regex=True)
            .replace("", float("nan"))
            .astype(float, errors="ignore")
        )
    return df


@st.cache_data
def load_funding_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.warning(f"Funding Rounds file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "Announced Date" in df.columns:
        df["Announced Date"] = pd.to_datetime(df["Announced Date"], errors="coerce")
        df.dropna(subset=["Announced Date"], inplace=True)
        df["Year Announced"] = df["Announced Date"].dt.year

    if "Money Raised" in df.columns:
        df["Money Raised"] = (
            df["Money Raised"]
            .replace(r"[^0-9.]", "", regex=True)
            .replace("", float("nan"))
            .astype(float, errors="ignore")
        )
    else:
        df["Money Raised"] = float("nan")

    if "Funding Stage" not in df.columns:
        df["Funding Stage"] = "Unknown"

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


################################################################################
#                         COLOR & THEME CONFIG                                 #
################################################################################

def gradient_palette(n, color1="#822723", color2="#545465"):
    c1 = mcolors.to_rgb(color1)
    c2 = mcolors.to_rgb(color2)
    colors = [
        mcolors.to_hex(c) for c in np.linspace(c1, c2, n)
    ]
    return colors

sns.set_theme(style="whitegrid")

ANNOTATION_FONTSIZE = 7
AXIS_LABEL_FONTSIZE = 7
TICK_LABEL_FONTSIZE = 7

################################################################################
#                      MERGE & LOAD MAIN LOGIC                                 #
################################################################################

merged_territory_file = merge_territory_files()
territory_df = load_territory_data(merged_territory_file)

merged_funding_file = merge_funding_rounds_files()
funding_df = load_funding_data(merged_funding_file)

if territory_df.empty or territory_df["Year Founded"].isna().all():
    st.error("No valid 'Territory' data after merging. Please check your CSV files.")
    st.stop()

all_territory_years = territory_df["Year Founded"].dropna().unique()
all_funding_years = funding_df["Year Announced"].dropna().unique() if not funding_df.empty else []
combined_years = list(set(all_territory_years).union(set(all_funding_years)))
if not combined_years:
    st.error("No valid years found in the merged datasets.")
    st.stop()

min_year = int(min(combined_years))
max_year = int(max(combined_years))

################################################################################
#                           PAGE LAYOUT & HEADER                               #
################################################################################

LOGO_PATH = "tvf_logo.png"  # Adjust for your logo path

top_left, top_right = st.columns([0.8, 0.2])
with top_left:
    st.title("TVF Territory & Funding Dashboard")
with top_right:
    st.image(LOGO_PATH, use_container_width=True)

st.sidebar.title("Select Year Range")
selected_year_range = st.sidebar.slider(
    "Choose a range of years",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

filtered_territory = territory_df[
    territory_df["Year Founded"].between(selected_year_range[0], selected_year_range[1])
]

filtered_funding = funding_df[
    (funding_df["Year Announced"].between(selected_year_range[0], selected_year_range[1]))
    & (funding_df["Funding Stage"].isin(["Seed", "Early Stage Funding"]))
    ]

################################################################################
#                    TERRITORY DASHBOARD SECTIONS                              #
################################################################################

st.subheader("Territory Data")

# --- Metric Tiles ---
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

if "Funding Status" in filtered_territory.columns and not filtered_territory.empty:
    has_funding_count = filtered_territory["Funding Status"].dropna().shape[0]
else:
    has_funding_count = 0

tile_colors = ["#822723", "#822723", "#822723", "#822723"]

def colored_tile(column, label, value, bg_color):
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

# Charts for Territory
col_left, col_right = st.columns(2)
col_left.subheader("Startups by Year")

if filtered_territory.empty:
    col_left.warning("No territory data in this range.")
else:
    year_counts = filtered_territory["Year Founded"].value_counts().sort_index()
    if year_counts.empty:
        col_left.warning("No data found for selected range.")
    else:
        data_for_plot = pd.DataFrame({"Year": year_counts.index, "Count": year_counts.values})
        num_bars = data_for_plot.shape[0]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x="Year",
            y="Count",
            data=data_for_plot,
            palette=gradient_palette(num_bars),
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
                fontsize=ANNOTATION_FONTSIZE
            )
        ax.set_xlabel("Year Founded", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax.set_ylim(0, data_for_plot["Count"].max() * 1.1)
        col_left.pyplot(fig)

col_right.subheader("Top Industries")
if "Industry Groups" not in filtered_territory.columns or filtered_territory.empty:
    col_right.warning("No industry data in this range.")
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
        num_bars = top10_df.shape[0]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            y="Industry",
            x="Count",
            data=top10_df,
            palette=gradient_palette(num_bars),
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
                fontsize=ANNOTATION_FONTSIZE
            )
        ax2.set_xlabel("Count", fontsize=AXIS_LABEL_FONTSIZE)
        ax2.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)
        ax2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
        ax2.set_ylim(-0.5, len(top10_df) - 0.5)
        col_right.pyplot(fig2)

st.write("---")

################################################################################
#                     FUNDING ROUNDS DASHBOARD                                 #
################################################################################

st.subheader("Funding Rounds (Seed & Early)")

if filtered_funding.empty:
    st.warning("No seed/early rounds in the selected year range.")
else:
    total_rounds = filtered_funding.shape[0]

    if not filtered_funding["Money Raised"].dropna().empty:
        avg_money_val = filtered_funding["Money Raised"].dropna().mean()
        avg_money_str = f"{avg_money_val:,.0f} EUR"
    else:
        avg_money_str = "0 EUR"

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
            left_fr.warning("No valid 'Year Announced' data in range.")
        else:
            data_fr_plot = pd.DataFrame({"Year": year_counts_fr.index, "Rounds": year_counts_fr.values})
            num_bars = data_fr_plot.shape[0]
            fig_fr, ax_fr = plt.subplots(figsize=(5, 4))
            sns.barplot(
                x="Year",
                y="Rounds",
                data=data_fr_plot,
                palette=gradient_palette(num_bars),
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
        left_fr.warning("No 'Year Announced' column found in funding data.")

    # Investment by Year (millions)
    right_fr.subheader("Investment by Year")
    if "Year Announced" in filtered_funding.columns:
        sum_by_year = filtered_funding.groupby("Year Announced")["Money Raised"].sum().sort_index()
        if sum_by_year.empty:
            right_fr.warning("No valid 'Money Raised' data in range.")
        else:
            fig_sum, ax_sum = plt.subplots(figsize=(5, 4))
            x_values = sum_by_year.index.astype(int)
            y_values = sum_by_year.values
            num_bars = len(y_values)

            sns.barplot(
                x=x_values,
                y=y_values,
                palette=gradient_palette(num_bars),
                edgecolor="black",
                ax=ax_sum
            )

            for i, val in enumerate(y_values):
                text_pos = val + (val * 0.01 if val != 0 else 1)
                ann_str = f"{val / 1_000_000:.1f}M"
                ax_sum.text(
                    i,
                    text_pos,
                    ann_str,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=ANNOTATION_FONTSIZE
                )
            ax_sum.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
            ax_sum.set_ylabel("EUR (millions)", fontsize=AXIS_LABEL_FONTSIZE)
            ax_sum.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1_000_000:.0f}M"))
            ax_sum.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

            max_val = y_values.max()
            ax_sum.set_ylim(0, max_val * 1.15 if max_val != 0 else 1)
            right_fr.pyplot(fig_sum)
    else:
        right_fr.warning("No 'Year Announced' or 'Money Raised' column found in funding data.")

    st.write("---")

    # Locations & Recent Rounds
    loc_col, table_col = st.columns(2)

    loc_col.subheader("Top Locations")
    if ("Organization Location" not in filtered_funding.columns
            or filtered_funding["Organization Location"].dropna().empty):
        loc_col.warning("No location data in range.")
    else:
        loc_counts = filtered_funding["Organization Location"].value_counts().head(10)
        if loc_counts.empty:
            loc_col.warning("No locations found.")
        else:
            loc_df = pd.DataFrame({"Location": loc_counts.index, "Count": loc_counts.values})
            num_bars = loc_df.shape[0]
            fig_loc, ax_loc = plt.subplots(figsize=(6, 4))
            sns.barplot(
                y="Location",
                x="Count",
                data=loc_df,
                palette=gradient_palette(num_bars),
                edgecolor="black",
                ax=ax_loc
            )
            for i, row_ in loc_df.iterrows():
                ax_loc.text(
                    row_["Count"] + 0.3,
                    i,
                    str(int(row_["Count"])),
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
        table_col.warning("No 'Announced Date' column to show recent rounds.")

    st.write("---")

    ############################################################
    #             INVESTOR ANALYSIS (Seed/Early)               #
    ############################################################

    st.write("### Investor Analysis")
    if "Investor Names" not in filtered_funding.columns:
        st.warning("No 'Investor Names' column found in dataset.")
    else:
        all_investors = []
        for val in filtered_funding["Investor Names"].dropna():
            i_list = [inv.strip() for inv in val.split(",") if inv.strip()]
            all_investors.extend(i_list)

        if not all_investors:
            st.warning("No valid investor names found.")
        else:
            colInv1, colInv2 = st.columns(2)

            with colInv1:
                st.subheader("Top Investors (Count)")
                investor_counts = Counter(all_investors)
                top10 = investor_counts.most_common(10)
                df_top10_count = pd.DataFrame(top10, columns=["Investor", "Count"])
                num_bars = df_top10_count.shape[0]

                fig_i1, ax_i1 = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=df_top10_count,
                    y="Investor",
                    x="Count",
                    palette=gradient_palette(num_bars),
                    edgecolor="black",
                    ax=ax_i1
                )
                for i, row_ in df_top10_count.iterrows():
                    ax_i1.text(
                        row_["Count"] + 0.1,
                        i,
                        str(row_["Count"]),
                        va="center",
                        color="black",
                        fontsize=ANNOTATION_FONTSIZE
                    )
                ax_i1.set_xlabel("Rounds", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i1.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i1.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
                colInv1.pyplot(fig_i1)

            with colInv2:
                st.subheader("Top Investors (Sum of Total Round Sizes in EUR)")
                investor_money = defaultdict(float)

                for idx, row_ in filtered_funding.dropna(subset=["Investor Names"]).iterrows():
                    money_raised = row_["Money Raised"] if not pd.isna(row_["Money Raised"]) else 0.0
                    i_list = [inv.strip() for inv in row_["Investor Names"].split(",") if inv.strip()]
                    for inv in i_list:
                        investor_money[inv] += money_raised

                sorted_investors = sorted(investor_money.items(), key=lambda x: x[1], reverse=True)
                top10_by_money = sorted_investors[:10]
                df_top10_money = pd.DataFrame(top10_by_money, columns=["Investor", "TotalMoney"])
                num_bars = df_top10_money.shape[0]

                fig_i2, ax_i2 = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=df_top10_money,
                    x="Investor",
                    y="TotalMoney",
                    palette=gradient_palette(num_bars),
                    edgecolor="black",
                    ax=ax_i2
                )
                # Annotate each bar in millions
                for i, row_ in df_top10_money.iterrows():
                    val = row_["TotalMoney"]
                    text_pos = val + (val * 0.01 if val != 0 else 1)
                    ann_str = f"{val / 1_000_000:.1f}M"
                    ax_i2.text(
                        i,
                        text_pos,
                        ann_str,
                        ha="center",
                        va="bottom",
                        color="black",
                        fontsize=ANNOTATION_FONTSIZE
                    )

                ax_i2.set_xlabel("Investor", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i2.set_ylabel("EUR (millions)", fontsize=AXIS_LABEL_FONTSIZE)
                ax_i2.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, pos: f"{x / 1_000_000:.0f}M")
                )
                plt.setp(ax_i2.get_xticklabels(), rotation=45, ha="right")
                ax_i2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
                colInv2.pyplot(fig_i2)

st.sidebar.write("Use the year range slider to explore different time periods.")
