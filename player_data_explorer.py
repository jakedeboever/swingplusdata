import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Settings ---
FIXED_ORDER = [
    "year", "Team", "name", "Age", "side",
    "swing_plus", "xwobacon", "predicted_xwobacon", "xwobacon_diff"
]

def safe_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else ""

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("final kfold data.csv")

    # Always drop Player ID if it exists
    if "Player ID" in df.columns:
        df = df.drop(columns=["id"])

    # Enforce initial column order
    cols = list(df.columns)
    ordered = [c for c in FIXED_ORDER if c in cols] + [c for c in cols if c not in FIXED_ORDER]
    df = df[ordered]
    return df

df = load_data()

st.title("MLB Player & Team Data Explorer")

# --- Player Search Bar (no multiselect filter) ---
all_players = sorted(df["name"].dropna().unique())
player_search = st.text_input("Search Player")
if player_search:
    selected_players = [p for p in all_players if player_search.lower() in p.lower()]
else:
    selected_players = all_players

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Team filter (still available for filtering only)
all_teams = sorted(df["Team"].dropna().unique()) if "Team" in df.columns else []
selected_teams = st.sidebar.multiselect("Select Teams", all_teams, default=all_teams)

# Year filter
all_years = sorted(df["year"].dropna().unique()) if "year" in df.columns else []
selected_years = st.sidebar.multiselect("Select Years", all_years, default=all_years)

# Aggregation mode: ONLY per-year rows or aggregate across selected years BY PLAYER
agg_mode = st.sidebar.radio(
    "View",
    ["Per Year (Player)", "Aggregate Selected Years (by Player)"],
    index=0
)

# --- Filter Data ---
mask = (
    df["name"].isin(selected_players)
    & df["Team"].isin(selected_teams)
    & df["year"].isin(selected_years)
)
filtered_df = df[mask].copy()

if agg_mode == "Aggregate Selected Years (by Player)":
    aggr_dict = {}

    if "year" in filtered_df.columns:
        aggr_dict["year"] = lambda x: ", ".join(map(str, sorted(pd.Series(x).dropna().unique())))
    if "Team" in filtered_df.columns:
        aggr_dict["Team"] = lambda x: ", ".join(sorted(pd.Series(x).dropna().unique()))
    if "Side" in filtered_df.columns:
        aggr_dict["Side"] = safe_mode
    if "age" in filtered_df.columns:
        aggr_dict["age"] = "mean"

    num_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if c not in ["year"]:
            aggr_dict[c] = "mean"

    aggr_df = filtered_df.groupby("name").agg(aggr_dict).reset_index()

    # --- Post-processing rounding ---
    for col in aggr_df.columns:
        if col.strip().lower() == "swing+":
            aggr_df[col] = aggr_df[col].round(0).astype(int)
        elif col in ["xwobacon", "Expected xwobacon", "xwobacon diff"]:
            aggr_df[col] = aggr_df[col].round(3)



    # Group by player name only
    group_df = filtered_df.groupby("name").agg(aggr_dict).reset_index()

    aggr_df = group_df
else:
    # Per-year rows, no aggregation
    aggr_df = filtered_df.copy()

# --- Always drop Player ID just in case (post-aggregation) ---
if "Player ID" in aggr_df.columns:
    aggr_df = aggr_df.drop(columns=["Player ID"])

# --- Enforce column order ALWAYS ---
cols_present = list(aggr_df.columns)
ordered_cols = [c for c in FIXED_ORDER if c in cols_present] + [c for c in cols_present if c not in FIXED_ORDER]
aggr_df = aggr_df[ordered_cols]

# --- Stat Filters (numeric only) ---
num_cols = aggr_df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
st.sidebar.subheader("Stat Filters")
for col in num_cols:
    min_val = float(aggr_df[col].min())
    max_val = float(aggr_df[col].max())
    sel_min, sel_max = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
    aggr_df = aggr_df[(aggr_df[col] >= sel_min) & (aggr_df[col] <= sel_max)]

# --- Display Data ---
st.subheader("Filtered Data")
st.dataframe(aggr_df)

# --- Download Data Button ---
st.download_button(
    label="Download Filtered Data as CSV",
    data=aggr_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# --- Plotting ---
if len(num_cols) >= 2:
    x_axis = st.selectbox("X-axis", num_cols, index=0)
    y_axis = st.selectbox("Y-axis", num_cols, index=1)

    if x_axis and y_axis:
        st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
        fig, ax = plt.subplots()
        if agg_mode == "Per Year (Player)" and "year" in aggr_df.columns:
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, hue="year", ax=ax)
        else:
            # In aggregated mode, 'year' is a comma-separated string, so no hue
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

        # Save plot to buffer for download
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Scatter Plot as PNG",
            data=buf,
            file_name="scatter_plot.png",
            mime="image/png"
        )

        # Correlation
        corr = aggr_df[[x_axis, y_axis]].corr().iloc[0, 1]
        st.write(f"**Correlation Coefficient (r):** {corr:.3f}")
else:
    st.write("Not enough numeric columns to plot.")
