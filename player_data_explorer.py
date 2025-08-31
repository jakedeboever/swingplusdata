import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# ---------- Config ----------
FIXED_ORDER = [
    "year", "Team", "name", "age", "Side",
    "Swing+", "xwobacon", "Expected xwobacon", "xwobacon diff"
]

# ---------- Helpers ----------
def normalize(s):
    """Normalize a string for fuzzy column matching."""
    if s is None:
        return ""
    s = str(s)
    # remove non-alphanumeric, lowercase
    return re.sub(r'[^0-9a-z]', '', s.lower())

def find_col(df, targets):
    """Return first dataframe column name that matches any target (normalized).
       targets: list of possible names to match (strings)."""
    if df is None or len(df.columns) == 0:
        return None
    col_norm = {col: normalize(col) for col in df.columns}
    target_norms = [normalize(t) for t in targets]
    # exact matches first
    for t in target_norms:
        for col, cn in col_norm.items():
            if cn == t:
                return col
    # substring matches next
    for t in target_norms:
        for col, cn in col_norm.items():
            if t in cn or cn in t:
                return col
    return None

def to_numeric_series(s):
    """Try to convert a Series to numeric aggressively (strip commas, % and parentheses)."""
    # convert to string, remove commas, percent signs, parentheses and whitespace
    cleaned = s.astype(str).str.replace(',', '', regex=False)\
                          .str.replace('%', '', regex=False)\
                          .str.replace('(', '', regex=False)\
                          .str.replace(')', '', regex=False)\
                          .str.strip()
    return pd.to_numeric(cleaned, errors='coerce')

def safe_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else pd.NA

# ---------- Load ----------
@st.cache_data
def load_data(path="final kfold data.csv"):
    df = pd.read_csv(path, dtype=object)  # read as object to avoid silent dtype surprises

    # Drop any Player ID-like columns
    pid = find_col(df, ["player id", "playerid", "player_id"])
    if pid:
        df = df.drop(columns=[pid])

    return df

df = load_data()

st.title("MLB Player & Team Data Explorer")

# ---------- Identify important columns (fuzzy) ----------
name_col = find_col(df, ["name", "playername", "player"])
year_col = find_col(df, ["year", "season"])
team_col = find_col(df, ["team"])
side_col = find_col(df, ["side", "hand", "bats", "throws"])
age_col = find_col(df, ["age"])
swing_col = find_col(df, ["swing+", "swingplus", "swing plus"])
xwobacon_col = find_col(df, ["xwobacon", "xwoba", "xwo bacon"])
expected_xwobacon_col = find_col(df, ["expected xwobacon", "expected_xwobacon", "expectedxwobacon"])
xwobacon_diff_col = find_col(df, ["xwobacon diff", "xwobacon_diff", "xwobacondiff", "xwobacon diff"])

# If name_col is missing we can't proceed sensibly
if name_col is None:
    st.error("Could not find a player name column in the CSV. Expected column like 'name'.")
    st.stop()

# ---------- Player search input (single text search) ----------
all_players = sorted(df[name_col].dropna().unique())
player_search = st.text_input("Search Player")
if player_search:
    selected_players = [p for p in all_players if player_search.lower() in str(p).lower()]
else:
    selected_players = all_players

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_teams = sorted(df[team_col].dropna().unique()) if team_col else []
selected_teams = st.sidebar.multiselect("Select Teams", all_teams, default=all_teams)

all_years = sorted(df[year_col].dropna().unique()) if year_col else []
selected_years = st.sidebar.multiselect("Select Years", all_years, default=all_years)

agg_mode = st.sidebar.radio("View", ["Per Year (Player)", "Aggregate Selected Years (by Player)"], index=0)

# ---------- Filter df ----------
mask = df[name_col].isin(selected_players)
if team_col:
    mask &= df[team_col].isin(selected_teams)
if year_col:
    mask &= df[year_col].isin(selected_years)
filtered_df = df[mask].copy()

# ---------- Coerce key numeric columns BEFORE aggregation ----------
# This helps ensure means/rounding work even if columns were strings
for c in [swing_col, xwobacon_col, expected_xwobacon_col, xwobacon_diff_col, age_col]:
    if c and c in filtered_df.columns:
        filtered_df[c] = to_numeric_series(filtered_df[c])

# ---------- Aggregation (only across years by player) ----------
if agg_mode == "Aggregate Selected Years (by Player)":
    # Build safe agg dict
    aggr_dict = {}

    # non-numeric rollups
    if year_col and year_col in filtered_df.columns:
        aggr_dict[year_col] = lambda x: ", ".join(map(str, sorted(pd.Series(x).dropna().unique())))
    if team_col and team_col in filtered_df.columns:
        aggr_dict[team_col] = lambda x: ", ".join(map(str, sorted(pd.Series(x).dropna().unique())))
    if side_col and side_col in filtered_df.columns:
        aggr_dict[side_col] = safe_mode
    if age_col and age_col in filtered_df.columns:
        aggr_dict[age_col] = "mean"

    # For all columns that look numeric (after coercion), apply mean
    # If a column couldn't be coerced earlier, this will silently skip
    numeric_candidates = filtered_df.columns[filtered_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notna().sum() > 0)]
    for c in numeric_candidates:
        if c not in aggr_dict and c != name_col:
            aggr_dict[c] = "mean"

    # Always group by the original name column
    aggr_df = filtered_df.groupby(name_col).agg(aggr_dict).reset_index()

else:
    aggr_df = filtered_df.copy()

# ---------- Post-aggregation coercion & rounding ----------
# Ensure relevant columns are numeric
for c in [swing_col, xwobacon_col, expected_xwobacon_col, xwobacon_diff_col, age_col]:
    if c and c in aggr_df.columns:
        aggr_df[c] = to_numeric_series(aggr_df[c])

# Round Swing+ to whole numbers (use nullable Int to tolerate NaNs)
if swing_col and swing_col in aggr_df.columns:
    # rounding might produce floats like 105.0 — convert safely to Int64
    aggr_df[swing_col] = aggr_df[swing_col].round(0)
    # convert to pandas nullable integer (preserves NA)
    try:
        aggr_df[swing_col] = aggr_df[swing_col].astype("Int64")
    except Exception:
        # fallback: keep as int where possible
        aggr_df[swing_col] = pd.to_numeric(aggr_df[swing_col], errors='coerce')

# Round xwobacon stats to 3 decimals
for col in [xwobacon_col, expected_xwobacon_col, xwobacon_diff_col]:
    if col and col in aggr_df.columns:
        aggr_df[col] = pd.to_numeric(aggr_df[col], errors='coerce').round(3)

# ---------- Always drop Player ID post-aggregation if present ----------
pid_after = find_col(aggr_df, ["player id", "playerid", "player_id"])
if pid_after:
    aggr_df = aggr_df.drop(columns=[pid_after])

# ---------- Rename matched columns to your exact labels and enforce order ----------
rename_map = {}
# map found columns to the canonical FIXED_ORDER labels
for desired in FIXED_ORDER:
    found = find_col(aggr_df, [desired])
    if found:
        rename_map[found] = desired

if rename_map:
    aggr_df = aggr_df.rename(columns=rename_map)

# Ensure all FIXED_ORDER columns exist (create if missing)
for desired in FIXED_ORDER:
    if desired not in aggr_df.columns:
        aggr_df[desired] = pd.NA

# Reorder: fixed first, then any remaining columns
ordered_cols = [c for c in FIXED_ORDER] + [c for c in aggr_df.columns if c not in FIXED_ORDER]
aggr_df = aggr_df[ordered_cols]

# ---------- Stat Filters (numeric only) ----------
num_cols = aggr_df.select_dtypes(include=["number"]).columns.tolist()
st.sidebar.subheader("Stat Filters")
for col in num_cols:
    min_val = float(aggr_df[col].min(skipna=True)) if not aggr_df[col].isna().all() else 0.0
    max_val = float(aggr_df[col].max(skipna=True)) if not aggr_df[col].isna().all() else 1.0
    sel_min, sel_max = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
    aggr_df = aggr_df[(aggr_df[col] >= sel_min) & (aggr_df[col] <= sel_max)]

# ---------- Display ----------
st.subheader("Filtered Data")
st.dataframe(aggr_df)

# ---------- Download ----------
st.download_button(
    label="Download Filtered Data as CSV",
    data=aggr_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_data.csv",
    mime="text/csv"
)

# ---------- Plotting ----------
if len(num_cols) >= 2:
    x_axis = st.selectbox("X-axis", num_cols, index=0)
    y_axis = st.selectbox("Y-axis", num_cols, index=1)

    if x_axis and y_axis:
        st.subheader(f"Scatter Plot: {x_axis} vs {y_axis}")
        fig, ax = plt.subplots()
        if agg_mode == "Per Year (Player)" and "year" in aggr_df.columns:
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, hue="year", ax=ax)
        else:
            sns.scatterplot(data=aggr_df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Scatter Plot as PNG",
            data=buf,
            file_name="scatter_plot.png",
            mime="image/png"
        )

        corr = aggr_df[[x_axis, y_axis]].corr().iloc[0, 1]
        st.write(f"**Correlation Coefficient (r):** {corr:.3f}")
else:
    st.write("Not enough numeric columns to plot.")
